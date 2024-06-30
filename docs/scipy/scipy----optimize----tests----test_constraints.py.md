# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_constraints.py`

```
import pytest  # 导入 pytest 模块，用于测试
import numpy as np  # 导入 NumPy 库并重命名为 np
from numpy.testing import TestCase, assert_array_equal  # 从 NumPy 的测试模块中导入 TestCase 类和 assert_array_equal 函数
import scipy.sparse as sps  # 导入 SciPy 稀疏矩阵模块并重命名为 sps
from scipy.optimize._constraints import (  # 从 SciPy 优化模块中导入多个约束类和函数
    Bounds, LinearConstraint, NonlinearConstraint, PreparedConstraint,
    new_bounds_to_old, old_bound_to_new, strict_bounds)

class TestStrictBounds(TestCase):  # 定义测试类 TestStrictBounds，继承自 TestCase 类

    def test_scalarvalue_unique_enforce_feasibility(self):  # 定义测试方法 test_scalarvalue_unique_enforce_feasibility
        m = 3  # 设置变量 m 的值为 3
        lb = 2  # 设置变量 lb 的值为 2
        ub = 4  # 设置变量 ub 的值为 4
        enforce_feasibility = False  # 设置变量 enforce_feasibility 的值为 False
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])  # 断言 strict_lb 应该等于 [-∞, -∞, -∞]
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])  # 断言 strict_ub 应该等于 [∞, ∞, ∞]

        enforce_feasibility = True  # 将 enforce_feasibility 设为 True
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 再次调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [2, 2, 2])  # 断言 strict_lb 应该等于 [2, 2, 2]
        assert_array_equal(strict_ub, [4, 4, 4])  # 断言 strict_ub 应该等于 [4, 4, 4]

    def test_vectorvalue_unique_enforce_feasibility(self):  # 定义测试方法 test_vectorvalue_unique_enforce_feasibility
        m = 3  # 设置变量 m 的值为 3
        lb = [1, 2, 3]  # 设置变量 lb 的值为 [1, 2, 3]
        ub = [4, 5, 6]  # 设置变量 ub 的值为 [4, 5, 6]
        enforce_feasibility = False  # 设置变量 enforce_feasibility 的值为 False
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])  # 断言 strict_lb 应该等于 [-∞, -∞, -∞]
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])  # 断言 strict_ub 应该等于 [∞, ∞, ∞]

        enforce_feasibility = True  # 将 enforce_feasibility 设为 True
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 再次调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [1, 2, 3])  # 断言 strict_lb 应该等于 [1, 2, 3]
        assert_array_equal(strict_ub, [4, 5, 6])  # 断言 strict_ub 应该等于 [4, 5, 6]

    def test_scalarvalue_vector_enforce_feasibility(self):  # 定义测试方法 test_scalarvalue_vector_enforce_feasibility
        m = 3  # 设置变量 m 的值为 3
        lb = 2  # 设置变量 lb 的值为 2
        ub = 4  # 设置变量 ub 的值为 4
        enforce_feasibility = [False, True, False]  # 设置变量 enforce_feasibility 的值为 [False, True, False]
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [-np.inf, 2, -np.inf])  # 断言 strict_lb 应该等于 [-∞, 2, -∞]
        assert_array_equal(strict_ub, [np.inf, 4, np.inf])  # 断言 strict_ub 应该等于 [∞, 4, ∞]

    def test_vectorvalue_vector_enforce_feasibility(self):  # 定义测试方法 test_vectorvalue_vector_enforce_feasibility
        m = 3  # 设置变量 m 的值为 3
        lb = [1, 2, 3]  # 设置变量 lb 的值为 [1, 2, 3]
        ub = [4, 6, np.inf]  # 设置变量 ub 的值为 [4, 6, ∞]
        enforce_feasibility = [True, False, True]  # 设置变量 enforce_feasibility 的值为 [True, False, True]
        strict_lb, strict_ub = strict_bounds(lb, ub,  # 调用 strict_bounds 函数，计算 strict_lb 和 strict_ub
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [1, -np.inf, 3])  # 断言 strict_lb 应该等于 [1, -∞, 3]
        assert_array_equal(strict_ub, [4, np.inf, np.inf])  # 断言 strict_ub 应该等于 [4, ∞, ∞]


def test_prepare_constraint_infeasible_x0():  # 定义测试方法 test_prepare_constraint_infeasible_x0
    lb = np.array([0, 20, 30])  # 创建 NumPy 数组 lb，其值为 [0, 20, 30]
    ub = np.array([0.5, np.inf, 70])  # 创建 NumPy 数组 ub，其值为 [0.5, ∞, 70]
    x0 = np.array([1, 2, 3])  # 创建 NumPy 数组 x0，其值为 [1, 2, 3]
    enforce_feasibility = np.array([False, True, True], dtype=bool)  # 创建布尔型 NumPy 数组 enforce_feasibility，其值为 [False, True, True]
    bounds = Bounds(lb, ub, enforce_feasibility)  # 创建 Bounds 对象 bounds，用于表示变量的上下界和可行性约束
    # 使用 pytest 检查 PreparedConstraint 构造函数是否会引发 ValueError 异常
    pytest.raises(ValueError, PreparedConstraint, bounds, x0)

    # 创建一个 PreparedConstraint 实例 pc，使用 Bounds 对象和给定的约束条件
    pc = PreparedConstraint(Bounds(lb, ub), [1, 2, 3])
    # 断言 pc.violation([1, 2, 3]) 中是否有任何一个约束条件被违反
    assert (pc.violation([1, 2, 3]) > 0).any()
    # 断言 pc.violation([0.25, 21, 31]) 中是否所有约束条件都没有被违反
    assert (pc.violation([0.25, 21, 31]) == 0).all()

    # 定义一个 NumPy 数组 x0
    x0 = np.array([1, 2, 3, 4])
    # 定义一个 NumPy 数组 A
    A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
    # 定义一个布尔类型的 NumPy 数组 enforce_feasibility
    enforce_feasibility = np.array([True, True, True], dtype=bool)
    # 创建一个线性约束对象 linear，通过 LinearConstraint 类
    linear = LinearConstraint(A, -np.inf, 0, enforce_feasibility)
    # 使用 pytest 检查 PreparedConstraint 构造函数是否会引发 ValueError 异常
    pytest.raises(ValueError, PreparedConstraint, linear, x0)

    # 创建一个 PreparedConstraint 实例 pc，使用 LinearConstraint 对象和给定的约束条件
    pc = PreparedConstraint(LinearConstraint(A, -np.inf, 0),
                            [1, 2, 3, 4])
    # 断言 pc.violation([1, 2, 3, 4]) 中是否有任何一个约束条件被违反
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    # 断言 pc.violation([-10, 2, -10, 4]) 中是否所有约束条件都没有被违反
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()

    # 定义一个函数 fun，返回 A 矩阵与输入向量 x 的乘积
    def fun(x):
        return A.dot(x)

    # 定义一个函数 jac，返回 A 矩阵本身
    def jac(x):
        return A

    # 定义一个函数 hess，返回一个稀疏矩阵，这里是一个空的 4x4 稀疏矩阵
    def hess(x, v):
        return sps.csr_matrix((4, 4))

    # 创建一个非线性约束对象 nonlinear，通过 NonlinearConstraint 类
    nonlinear = NonlinearConstraint(fun, -np.inf, 0, jac, hess,
                                    enforce_feasibility)
    # 使用 pytest 检查 PreparedConstraint 构造函数是否会引发 ValueError 异常
    pytest.raises(ValueError, PreparedConstraint, nonlinear, x0)

    # 创建一个 PreparedConstraint 实例 pc，使用 NonlinearConstraint 对象和给定的约束条件
    pc = PreparedConstraint(nonlinear, [-10, 2, -10, 4])
    # 断言 pc.violation([1, 2, 3, 4]) 中是否有任何一个约束条件被违反
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    # 断言 pc.violation([-10, 2, -10, 4]) 中是否所有约束条件都没有被违反
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()
def test_violation():
    # 定义一个非线性约束函数 cons_f，接受一个向量 x，并返回一个包含两个元素的 NumPy 数组
    def cons_f(x):
        return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])

    # 创建一个非线性约束对象 nlc，限制 cons_f 的输出在 [-1, -0.85] 和 [2, 2] 之间
    nlc = NonlinearConstraint(cons_f, [-1, -0.8500], [2, 2])
    
    # 创建一个预处理约束对象 pc，将 nlc 作为其约束条件，[0.5, 1] 作为输入
    pc = PreparedConstraint(nlc, [0.5, 1])
    
    # 断言检查 pc.violation([0.5, 1]) 返回的结果与预期的 [0., 0.] 相等
    assert_array_equal(pc.violation([0.5, 1]), [0., 0.])

    # 使用 np.testing.assert_almost_equal 检查 pc.violation([0.5, 1.2]) 的结果接近于 [0., 0.1]
    np.testing.assert_almost_equal(pc.violation([0.5, 1.2]), [0., 0.1])

    # 使用 np.testing.assert_almost_equal 检查 pc.violation([1.2, 1.2]) 的结果接近于 [0.64, 0]
    np.testing.assert_almost_equal(pc.violation([1.2, 1.2]), [0.64, 0])

    # 使用 np.testing.assert_almost_equal 检查 pc.violation([0.1, -1.2]) 的结果接近于 [0.19, 0]
    np.testing.assert_almost_equal(pc.violation([0.1, -1.2]), [0.19, 0])

    # 使用 np.testing.assert_almost_equal 检查 pc.violation([0.1, 2]) 的结果接近于 [0.01, 1.14]
    np.testing.assert_almost_equal(pc.violation([0.1, 2]), [0.01, 1.14])


def test_new_bounds_to_old():
    # 定义下界 lb 和上界 ub
    lb = np.array([-np.inf, 2, 3])
    ub = np.array([3, np.inf, 10])

    # 使用 new_bounds_to_old 将 lb 和 ub 转换为 bounds，并与预期的 bounds 进行断言
    bounds = [(None, 3), (2, None), (3, 10)]
    assert_array_equal(new_bounds_to_old(lb, ub, 3), bounds)

    # 类似地，对于单个 lb 的情况，进行断言
    bounds_single_lb = [(-1, 3), (-1, None), (-1, 10)]
    assert_array_equal(new_bounds_to_old(-1, ub, 3), bounds_single_lb)

    # 对于没有 lb 的情况，进行断言
    bounds_no_lb = [(None, 3), (None, None), (None, 10)]
    assert_array_equal(new_bounds_to_old(-np.inf, ub, 3), bounds_no_lb)

    # 对于单个 ub 的情况，进行断言
    bounds_single_ub = [(None, 20), (2, 20), (3, 20)]
    assert_array_equal(new_bounds_to_old(lb, 20, 3), bounds_single_ub)

    # 对于没有 ub 的情况，进行断言
    bounds_no_ub = [(None, None), (2, None), (3, None)]
    assert_array_equal(new_bounds_to_old(lb, np.inf, 3), bounds_no_ub)

    # 对于同时没有 lb 和 ub 的情况，进行断言
    bounds_single_both = [(1, 2), (1, 2), (1, 2)]
    assert_array_equal(new_bounds_to_old(1, 2, 3), bounds_single_both)

    # 对于完全没有 lb 和 ub 的情况，进行断言
    bounds_no_both = [(None, None), (None, None), (None, None)]
    assert_array_equal(new_bounds_to_old(-np.inf, np.inf, 3), bounds_no_both)


def test_old_bounds_to_new():
    # 定义 bounds
    bounds = ([1, 2], (None, 3), (-1, None))
    # 定义预期的 lb_true 和 ub_true
    lb_true = np.array([1, -np.inf, -1])
    ub_true = np.array([2, 3, np.inf])

    # 使用 old_bound_to_new 转换 bounds 为 lb 和 ub，并与预期的 lb_true 和 ub_true 进行断言
    lb, ub = old_bound_to_new(bounds)
    assert_array_equal(lb, lb_true)
    assert_array_equal(ub, ub_true)

    # 另一种情况下的断言
    bounds = [(-np.inf, np.inf), (np.array([1]), np.array([1]))]
    lb, ub = old_bound_to_new(bounds)

    assert_array_equal(lb, [-np.inf, 1])
    assert_array_equal(ub, [np.inf, 1])


class TestBounds:
    def test_repr(self):
        # 为了确保 eval 能够正常工作，导入必要的库
        from numpy import array, inf  # noqa: F401
        # 对多种情况下的 Bounds 对象进行断言
        for args in (
            (-1.0, 5.0),
            (-1.0, np.inf, True),
            (np.array([1.0, -np.inf]), np.array([2.0, np.inf])),
            (np.array([1.0, -np.inf]), np.array([2.0, np.inf]),
             np.array([True, False])),
        ):
            bounds = Bounds(*args)
            # 使用 eval(repr(Bounds(*args))) 生成新的 Bounds 对象，并与原始对象进行断言
            bounds2 = eval(repr(Bounds(*args)))
            assert_array_equal(bounds.lb, bounds2.lb)
            assert_array_equal(bounds.ub, bounds2.ub)
            assert_array_equal(bounds.keep_feasible, bounds2.keep_feasible)

    def test_array(self):
        # 测试 Bounds 对象是否能正确处理数组作为参数的情况
        # gh13501
        b = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])
        assert isinstance(b.lb, np.ndarray)
        assert isinstance(b.ub, np.ndarray)
    # 定义测试函数，验证默认参数设置是否正确
    def test_defaults(self):
        # 创建两个 Bounds 对象，一个使用默认参数，另一个指定范围为负无穷到正无穷的数组
        b1 = Bounds()
        b2 = Bounds(np.asarray(-np.inf), np.asarray(np.inf))
        # 断言默认下界与上界相等
        assert b1.lb == b2.lb
        # 断言默认上界与上界相等
        assert b1.ub == b2.ub

    # 定义测试函数，验证输入参数的有效性检查
    def test_input_validation(self):
        # 准备错误消息
        message = "Lower and upper bounds must be dense arrays."
        # 使用 pytest 的断言检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=message):
            Bounds(sps.coo_array([1, 2]), [1, 2])
        # 使用 pytest 的断言检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=message):
            Bounds([1, 2], sps.coo_array([1, 2]))

        # 准备错误消息
        message = "`keep_feasible` must be a dense array."
        # 使用 pytest 的断言检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=message):
            Bounds([1, 2], [1, 2], keep_feasible=sps.coo_array([True, True]))

        # 准备错误消息
        message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
        # 使用 pytest 的断言检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=message):
            Bounds([1, 2], [1, 2, 3])

    # 定义测试函数，验证 Bounds 对象的 residual 方法
    def test_residual(self):
        # 创建一个 Bounds 对象，设置下界为 -2，上界为 4
        bounds = Bounds(-2, 4)
        # 准备初始值数组
        x0 = [-1, 2]
        # 使用 numpy.testing.assert_allclose 断言 Bounds 对象的 residual 方法计算结果是否与预期接近
        np.testing.assert_allclose(bounds.residual(x0), ([1, 4], [5, 2]))
# 定义一个测试类 TestLinearConstraint，用于测试 LinearConstraint 类的功能
class TestLinearConstraint:

    # 测试默认参数情况下的 LinearConstraint 类
    def test_defaults(self):
        # 创建一个 4x4 的单位矩阵 A
        A = np.eye(4)
        # 使用单位矩阵 A 创建一个 LinearConstraint 对象 lc，没有指定上下界，默认为 (-inf, inf)
        lc = LinearConstraint(A)
        # 使用单位矩阵 A 创建另一个 LinearConstraint 对象 lc2，指定了上下界为 (-inf, inf)
        lc2 = LinearConstraint(A, -np.inf, np.inf)
        # 断言两个对象的下界 lb 和上界 ub 相等
        assert_array_equal(lc.lb, lc2.lb)
        assert_array_equal(lc.ub, lc2.ub)

    # 测试输入验证功能
    def test_input_validation(self):
        # 创建一个 4x4 的单位矩阵 A
        A = np.eye(4)

        # 测试当 lb 和 ub 的长度不匹配时，是否抛出 ValueError 异常
        message = "`lb`, `ub`, and `keep_feasible` must be broadcastable"
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, [1, 2], [1, 2, 3])

        # 测试当 lb 或 ub 不是密集数组时，是否抛出 ValueError 异常
        message = "Constraint limits must be dense arrays"
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, sps.coo_array([1, 2]), [2, 3])
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A, [1, 2], sps.coo_array([2, 3]))

        # 测试当 keep_feasible 不是密集数组时，是否抛出 ValueError 异常
        message = "`keep_feasible` must be a dense array"
        with pytest.raises(ValueError, match=message):
            keep_feasible = sps.coo_array([True, True])
            LinearConstraint(A, [1, 2], [2, 3], keep_feasible=keep_feasible)

        # 测试当 A 不是二维数组时，是否抛出 ValueError 异常
        A = np.empty((4, 3, 5))
        message = "`A` must have exactly two dimensions."
        with pytest.raises(ValueError, match=message):
            LinearConstraint(A)

    # 测试残差计算功能
    def test_residual(self):
        # 创建一个 2x2 的单位矩阵 A
        A = np.eye(2)
        # 使用单位矩阵 A 创建一个 LinearConstraint 对象 lc，指定上下界为 -2 和 4
        lc = LinearConstraint(A, -2, 4)
        # 定义一个初始向量 x0
        x0 = [-1, 2]
        # 断言计算得到的残差值与预期值在数值上非常接近
        np.testing.assert_allclose(lc.residual(x0), ([1, 4], [5, 2]))
```