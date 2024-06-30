# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__numdiff.py`

```
import math  # 导入数学模块
from itertools import product  # 导入 product 函数

import numpy as np  # 导入 NumPy 库
from numpy.testing import assert_allclose, assert_equal, assert_  # 导入 NumPy 测试工具
from pytest import raises as assert_raises  # 导入 pytest 的 raises 别名为 assert_raises

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix  # 导入稀疏矩阵类型

from scipy.optimize._numdiff import (
    _adjust_scheme_to_bounds, approx_derivative, check_derivative,
    group_columns, _eps_for_method, _compute_absolute_step)  # 导入数值微分相关函数


def test_group_columns():
    structure = [  # 定义一个二维数组作为矩阵结构
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ]
    for transform in [np.asarray, csr_matrix, csc_matrix, lil_matrix]:  # 遍历不同的矩阵转换方式
        A = transform(structure)  # 应用当前转换方式，将结构转换为矩阵
        order = np.arange(6)  # 创建一个包含 0 到 5 的数组
        groups_true = np.array([0, 1, 2, 0, 1, 2])  # 预期的列分组结果
        groups = group_columns(A, order)  # 使用给定的顺序进行列分组
        assert_equal(groups, groups_true)  # 断言分组结果与预期结果相等

        order = [1, 2, 4, 3, 5, 0]  # 指定另一种顺序
        groups_true = np.array([2, 0, 1, 2, 0, 1])  # 预期的列分组结果
        groups = group_columns(A, order)  # 使用指定顺序进行列分组
        assert_equal(groups, groups_true)  # 断言分组结果与预期结果相等

    # 测试列分组函数的重复性
    groups_1 = group_columns(A)  # 使用默认顺序进行列分组
    groups_2 = group_columns(A)  # 再次使用默认顺序进行列分组
    assert_equal(groups_1, groups_2)  # 断言两次分组结果相等


def test_correct_fp_eps():
    # 检查浮点数大小的相对步长是否正确
    EPS = np.finfo(np.float64).eps  # 获取 np.float64 类型的机器精度
    relative_step = {"2-point": EPS**0.5,  # 不同数值微分方法的相对步长
                    "3-point": EPS**(1/3),
                     "cs": EPS**0.5}
    for method in ['2-point', '3-point', 'cs']:  # 遍历不同的数值微分方法
        assert_allclose(
            _eps_for_method(np.float64, np.float64, method),  # 断言相对步长与预期相等
            relative_step[method])
        assert_allclose(
            _eps_for_method(np.complex128, np.complex128, method),  # 复数类型的相对步长断言
            relative_step[method]
        )

    # 检查另一种浮点数大小
    EPS = np.finfo(np.float32).eps  # 获取 np.float32 类型的机器精度
    relative_step = {"2-point": EPS**0.5,  # 不同数值微分方法的相对步长
                    "3-point": EPS**(1/3),
                     "cs": EPS**0.5}

    for method in ['2-point', '3-point', 'cs']:  # 遍历不同的数值微分方法
        assert_allclose(
            _eps_for_method(np.float64, np.float32, method),  # 断言相对步长与预期相等
            relative_step[method]
        )
        assert_allclose(
            _eps_for_method(np.float32, np.float64, method),  # 断言相对步长与预期相等
            relative_step[method]
        )
        assert_allclose(
            _eps_for_method(np.float32, np.float32, method),  # 断言相对步长与预期相等
            relative_step[method]
        )


class TestAdjustSchemeToBounds:  # 定义一个测试类，测试调整方案到边界
    # 测试无界限情况下的边界调整函数
    def test_no_bounds(self):
        # 创建一个包含三个零的数组
        x0 = np.zeros(3)
        # 创建一个包含三个1e-2值的数组
        h = np.full(3, 1e-2)
        # 创建一个与x0相同形状的空数组
        inf_lower = np.empty_like(x0)
        # 创建一个与x0相同形状的空数组
        inf_upper = np.empty_like(x0)
        # 将inf_lower数组填充为负无穷
        inf_lower.fill(-np.inf)
        # 将inf_upper数组填充为正无穷
        inf_upper.fill(np.inf)

        # 调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '1-sided', inf_lower, inf_upper)
        # 断言调整后的步长与原始步长相等
        assert_allclose(h_adjusted, h)
        # 断言所有的单边标志都为真
        assert_(np.all(one_sided))

        # 再次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '1-sided', inf_lower, inf_upper)
        # 断言调整后的步长与原始步长相等
        assert_allclose(h_adjusted, h)
        # 断言所有的单边标志都为真
        assert_(np.all(one_sided))

        # 再次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', inf_lower, inf_upper)
        # 断言调整后的步长与原始步长相等
        assert_allclose(h_adjusted, h)
        # 断言所有的单边标志都为假
        assert_(np.all(~one_sided))

        # 最后一次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', inf_lower, inf_upper)
        # 断言调整后的步长与原始步长相等
        assert_allclose(h_adjusted, h)
        # 断言所有的单边标志都为假
        assert_(np.all(~one_sided))

    # 测试有界限情况下的边界调整函数
    def test_with_bound(self):
        # 创建一个包含三个元素的数组，作为初始点x0
        x0 = np.array([0.0, 0.85, -0.85])
        # 创建一个包含三个-1的数组，作为下界lb
        lb = -np.ones(3)
        # 创建一个包含三个1的数组，作为上界ub
        ub = np.ones(3)
        # 创建一个包含三个元素的数组，作为步长h
        h = np.array([1, 1, -1]) * 1e-1

        # 调用边界调整函数，获取调整后的步长和不使用的返回值
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        # 断言调整后的步长与原始步长相等
        assert_allclose(h_adjusted, h)

        # 再次调用边界调整函数，获取调整后的步长和不使用的返回值
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([1, -1, 1]) * 1e-1)

        # 再次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.abs(h))
        # 断言所有的单边标志都为假
        assert_(np.all(~one_sided))

        # 最后一次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([1, -1, 1]) * 1e-1)
        # 断言单边标志与指定数组相等
        assert_equal(one_sided, np.array([False, True, True]))

    # 测试紧密边界情况下的边界调整函数
    def test_tight_bounds(self):
        # 创建一个包含两个元素的数组，作为下界lb
        lb = np.array([-0.03, -0.03])
        # 创建一个包含两个元素的数组，作为上界ub
        ub = np.array([0.05, 0.05])
        # 创建一个包含两个元素的数组，作为初始点x0
        x0 = np.array([0.0, 0.03])
        # 创建一个包含两个元素的数组，作为步长h
        h = np.array([-0.1, -0.1])

        # 调用边界调整函数，获取调整后的步长和不使用的返回值
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([0.05, -0.06]))

        # 再次调用边界调整函数，获取调整后的步长和不使用的返回值
        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([0.025, -0.03]))

        # 再次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([0.03, -0.03]))
        # 断言单边标志与指定数组相等
        assert_equal(one_sided, np.array([False, True]))

        # 最后一次调用边界调整函数，获取调整后的步长和是否单边标志
        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', lb, ub)
        # 断言调整后的步长与指定值相等
        assert_allclose(h_adjusted, np.array([0.015, -0.015]))
        # 断言单边标志与指定数组相等
        assert_equal(one_sided, np.array([False, True]))
class TestApproxDerivativesDense:
    # 定义测试类 TestApproxDerivativesDense
    def fun_scalar_scalar(self, x):
        # 定义函数 fun_scalar_scalar，返回 x 的双曲正弦值
        return np.sinh(x)

    def jac_scalar_scalar(self, x):
        # 定义函数 jac_scalar_scalar，返回 x 的双曲余弦值
        return np.cosh(x)

    def fun_scalar_vector(self, x):
        # 定义函数 fun_scalar_vector，返回一个包含三个元素的 numpy 数组
        # 元素分别为 x[0] 的平方、x[0] 的正切、e 的 x[0] 次幂
        return np.array([x[0]**2, np.tan(x[0]), np.exp(x[0])])

    def jac_scalar_vector(self, x):
        # 定义函数 jac_scalar_vector，返回一个形状为 (-1, 1) 的 numpy 数组
        # 其中包含三个元素：2 * x[0]、cos(x[0]) 的 -2 次幂、e 的 x[0] 次幂
        return np.array(
            [2 * x[0], np.cos(x[0]) ** -2, np.exp(x[0])]).reshape(-1, 1)

    def fun_vector_scalar(self, x):
        # 定义函数 fun_vector_scalar，返回 x[0] * x[1] 的正弦值乘以 log(x[0]) 的结果
        return np.sin(x[0] * x[1]) * np.log(x[0])

    def wrong_dimensions_fun(self, x):
        # 定义函数 wrong_dimensions_fun，返回一个包含三个元素的 numpy 数组
        # 元素分别为 x 的平方、x 的正切、e 的 x 次幂
        return np.array([x**2, np.tan(x), np.exp(x)])

    def jac_vector_scalar(self, x):
        # 定义函数 jac_vector_scalar，返回一个包含两个元素的 numpy 数组
        # 第一个元素为 x[1] * cos(x[0] * x[1]) * log(x[0]) + sin(x[0] * x[1]) / x[0]
        # 第二个元素为 x[0] * cos(x[0] * x[1]) * log(x[0])
        return np.array([
            x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) +
            np.sin(x[0] * x[1]) / x[0],
            x[0] * np.cos(x[0] * x[1]) * np.log(x[0])
        ])

    def fun_vector_vector(self, x):
        # 定义函数 fun_vector_vector，返回一个包含三个元素的 numpy 数组
        # 分别为 x[0] * sin(x[1])、x[1] * cos(x[0])、x[0] 的立方 * x[1] 的 -0.5 次幂
        return np.array([
            x[0] * np.sin(x[1]),
            x[1] * np.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def jac_vector_vector(self, x):
        # 定义函数 jac_vector_vector，返回一个包含三个元素的列表
        # 每个元素都是一个 numpy 数组，表示偏导数矩阵的一行
        return np.array([
            [np.sin(x[1]), x[0] * np.cos(x[1])],
            [-x[1] * np.sin(x[0]), np.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def fun_parametrized(self, x, c0, c1=1.0):
        # 定义函数 fun_parametrized，返回一个包含两个元素的 numpy 数组
        # 元素分别为 e 的 c0 * x[0] 次幂、e 的 c1 * x[1] 次幂
        return np.array([np.exp(c0 * x[0]), np.exp(c1 * x[1])])

    def jac_parametrized(self, x, c0, c1=0.1):
        # 定义函数 jac_parametrized，返回一个 2x2 的 numpy 数组
        # 元素为 [c0 * e 的 c0 * x[0] 次幂, 0] 和 [0, c1 * e 的 c1 * x[1] 次幂]
        return np.array([
            [c0 * np.exp(c0 * x[0]), 0],
            [0, c1 * np.exp(c1 * x[1])]
        ])

    def fun_with_nan(self, x):
        # 定义函数 fun_with_nan，如果 abs(x) <= 1e-8 则返回 x，否则返回 NaN
        return x if np.abs(x) <= 1e-8 else np.nan

    def jac_with_nan(self, x):
        # 定义函数 jac_with_nan，如果 abs(x) <= 1e-8 则返回 1.0，否则返回 NaN
        return 1.0 if np.abs(x) <= 1e-8 else np.nan

    def fun_zero_jacobian(self, x):
        # 定义函数 fun_zero_jacobian，返回一个包含两个元素的 numpy 数组
        # 元素分别为 x[0] * x[1] 和 cos(x[0] * x[1])
        return np.array([x[0] * x[1], np.cos(x[0] * x[1])])

    def jac_zero_jacobian(self, x):
        # 定义函数 jac_zero_jacobian，返回一个 2x2 的 numpy 数组
        # 元素为 [[x[1], x[0]], [-x[1] * sin(x[0] * x[1]), -x[0] * sin(x[0] * x[1])]]
        return np.array([
            [x[1], x[0]],
            [-x[1] * np.sin(x[0] * x[1]), -x[0] * np.sin(x[0] * x[1])]
        ])

    def jac_non_numpy(self, x):
        # 定义函数 jac_non_numpy，将 x 转换为标量后，返回 math.exp 的结果
        # 这里 x 可以是一个标量或者包含一个元素的数组 [val]
        xp = np.asarray(x).item()
        return math.exp(xp)

    def test_scalar_scalar(self):
        # 定义测试函数 test_scalar_scalar
        x0 = 1.0
        # 使用两点法计算 fun_scalar_scalar 在 x0 处的数值导数近似值
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point')
        # 使用默认方法（中心差分法）计算 fun_scalar_scalar 在 x0 处的数值导数近似值
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0)
        # 使用复数步长法计算 fun_scalar_scalar 在 x0 处的数值导数近似值
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs')
        # 计算 fun_scalar_scalar 在 x0 处的真实导数值
        jac_true = self.jac_scalar_scalar(x0)
        # 断言近似导数值 jac_diff_2 与真实导数值 jac_true 的误差在相对误差 1e-6 内
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言近似导数值 jac_diff_3 与真实导数值 jac_true 的误差在相对误差 1e-9 内
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        # 断言近似导数值 jac_diff_4 与真实导数值 jac_true 的误差在相对误差 1e-12 内
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
    def test_scalar_scalar_abs_step(self):
        # 测试对于标量到标量函数，使用不同的数值差分方法计算雅可比矩阵，并进行断言检查
        x0 = 1.0
        # 使用二点法计算数值导数，指定绝对步长为1.49e-8
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point', abs_step=1.49e-8)
        # 使用默认方法计算数值导数，指定绝对步长为1.49e-8
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0,
                                       abs_step=1.49e-8)
        # 使用复合辛普森法计算数值导数，指定绝对步长为1.49e-8
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs', abs_step=1.49e-8)
        # 计算真实的雅可比矩阵
        jac_true = self.jac_scalar_scalar(x0)
        # 断言二点法计算的数值导数与真实值的接近程度在相对误差1e-6内
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的数值导数与真实值的接近程度在相对误差1e-9内
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        # 断言复合辛普森法计算的数值导数与真实值的接近程度在相对误差1e-12内
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_scalar_vector(self):
        # 测试对于标量到向量函数，使用不同的数值差分方法计算雅可比矩阵，并进行断言检查
        x0 = 0.5
        # 使用二点法计算数值导数
        jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='2-point')
        # 使用默认方法计算数值导数
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0)
        # 使用复合辛普森法计算数值导数
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='cs')
        # 计算真实的雅可比矩阵
        jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
        # 断言二点法计算的数值导数与真实值的接近程度在相对误差1e-6内
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的数值导数与真实值的接近程度在相对误差1e-9内
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        # 断言复合辛普森法计算的数值导数与真实值的接近程度在相对误差1e-12内
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_scalar(self):
        # 测试对于向量到标量函数，使用不同的数值差分方法计算雅可比矩阵，并进行断言检查
        x0 = np.array([100.0, -0.5])
        # 使用二点法计算数值导数
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point')
        # 使用默认方法计算数值导数
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0)
        # 使用复合辛普森法计算数值导数
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs')
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_scalar(x0)
        # 断言二点法计算的数值导数与真实值的接近程度在相对误差1e-6内
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的数值导数与真实值的接近程度在相对误差1e-7内
        assert_allclose(jac_diff_3, jac_true, rtol=1e-7)
        # 断言复合辛普森法计算的数值导数与真实值的接近程度在相对误差1e-12内
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_scalar_abs_step(self):
        # 测试对于向量到标量函数，是否能够使用绝对步长进行数值导数计算，并进行断言检查
        x0 = np.array([100.0, -0.5])
        # 使用二点法计算数值导数，指定绝对步长为1.49e-8
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point', abs_step=1.49e-8)
        # 使用默认方法计算数值导数，指定绝对步长为1.49e-8，并且相对步长为无穷大（忽略相对步长）
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0,
                                       abs_step=1.49e-8, rel_step=np.inf)
        # 使用复合辛普森法计算数值导数，指定绝对步长为1.49e-8
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs', abs_step=1.49e-8)
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_scalar(x0)
        # 断言二点法计算的数值导数与真实值的接近程度在相对误差1e-6内
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的数值导数与真实值的接近程度在相对误差3e-9内
        assert_allclose(jac_diff_3, jac_true, rtol=3e-9)
        # 断言复合辛普森法计算的数值导数与真实值的接近程度在相对误差1e-12内
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)
    # 测试向量到向量函数的数值微分，使用不同的方法计算雅可比矩阵
    def test_vector_vector(self):
        # 定义输入向量 x0
        x0 = np.array([-100.0, 0.2])
        # 使用 '2-point' 方法计算数值微分的雅可比矩阵
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point')
        # 使用默认方法计算数值微分的雅可比矩阵
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0)
        # 使用 'cs' 方法计算数值微分的雅可比矩阵
        jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
                                       method='cs')
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_vector(x0)
        # 断言 '2-point' 方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_2, jac_true, rtol=1e-5)
        # 断言默认方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_3, jac_true, rtol=1e-6)
        # 断言 'cs' 方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    # 测试维度错误的情况
    def test_wrong_dimensions(self):
        # 定义错误的输入 x0
        x0 = 1.0
        # 断言运行时错误，因为传入了错误维度的函数
        assert_raises(RuntimeError, approx_derivative,
                      self.wrong_dimensions_fun, x0)
        # 调用传入至少为一维的 x0 的函数，以获取函数值 f0
        f0 = self.wrong_dimensions_fun(np.atleast_1d(x0))
        # 断言值错误，因为传入了不符合要求的 f0
        assert_raises(ValueError, approx_derivative,
                      self.wrong_dimensions_fun, x0, f0=f0)

    # 测试自定义相对步长的情况
    def test_custom_rel_step(self):
        # 定义输入向量 x0
        x0 = np.array([-0.1, 0.1])
        # 使用 '2-point' 方法和自定义相对步长计算数值微分的雅可比矩阵
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point', rel_step=1e-4)
        # 使用默认方法和自定义相对步长计算数值微分的雅可比矩阵
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
                                       rel_step=1e-4)
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_vector(x0)
        # 断言 '2-point' 方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_2, jac_true, rtol=1e-2)
        # 断言默认方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_3, jac_true, rtol=1e-4)

    # 测试带选项的情况
    def test_options(self):
        # 定义输入向量 x0 和常数 c0, c1
        x0 = np.array([1.0, 1.0])
        c0 = -1.0
        c1 = 1.0
        # 计算带参数化函数的初始函数值 f0
        f0 = self.fun_parametrized(x0, c0, c1=c1)
        # 计算真实的雅可比矩阵
        jac_true = self.jac_parametrized(x0, c0, c1)
        # 定义相对步长 rel_step
        rel_step = np.array([-1e-6, 1e-7])
        # 使用 '2-point' 方法、带选项的计算数值微分的雅可比矩阵
        jac_diff_2 = approx_derivative(
            self.fun_parametrized, x0, method='2-point', rel_step=rel_step,
            f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        # 使用默认方法、带选项的计算数值微分的雅可比矩阵
        jac_diff_3 = approx_derivative(
            self.fun_parametrized, x0, rel_step=rel_step,
            f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        # 断言 '2-point' 方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)

    # 测试带边界的 '2-point' 方法的情况
    def test_with_bounds_2_point(self):
        # 定义下界 lb 和上界 ub
        lb = -np.ones(2)
        ub = np.ones(2)

        # 定义输入向量 x0
        x0 = np.array([-2.0, 0.2])
        # 断言值错误，因为传入的 x0 超出了定义的边界
        assert_raises(ValueError, approx_derivative,
                      self.fun_vector_vector, x0, bounds=(lb, ub))

        # 定义输入向量 x0
        x0 = np.array([-1.0, 1.0])
        # 使用 '2-point' 方法和带边界的计算数值微分的雅可比矩阵
        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     method='2-point', bounds=(lb, ub))
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_vector(x0)
        # 断言 '2-point' 方法计算的雅可比矩阵与真实值的接近程度
        assert_allclose(jac_diff, jac_true, rtol=1e-6)
    # 定义一个测试函数，用于测试带有边界条件的三点数值求导方法

    def test_with_bounds_3_point(self):
        # 定义下界和上界
        lb = np.array([1.0, 1.0])
        ub = np.array([2.0, 2.0])

        # 定义初始点
        x0 = np.array([1.0, 2.0])
        
        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_vector(x0)

        # 使用数值逼近方法计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(self.fun_vector_vector, x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        # 使用指定的边界条件计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(lb, np.inf))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        # 使用另一组边界条件计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(-np.inf, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        # 使用另一组边界条件计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

    # 定义一个测试函数，用于测试紧密边界条件下的数值求导方法

    def test_tight_bounds(self):
        # 定义初始点
        x0 = np.array([10.0, 10.0])

        # 根据初始点定义边界范围
        lb = x0 - 3e-9
        ub = x0 + 2e-9

        # 计算真实的雅可比矩阵
        jac_true = self.jac_vector_vector(x0)

        # 使用指定的边界条件和方法计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

        # 使用指定的边界条件、方法和相对步长计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, method='2-point',
            rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

        # 使用指定的边界条件计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

        # 使用指定的边界条件、相对步长计算雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_true, jac_diff, rtol=1e-6)

    # 定义一个测试函数，用于测试边界条件切换时的数值求导方法

    def test_bound_switches(self):
        # 定义下界和上界
        lb = -1e-8
        ub = 1e-8
        
        # 定义初始点
        x0 = 0.0

        # 计算带 NaN 值的真实雅可比矩阵
        jac_true = self.jac_with_nan(x0)

        # 使用指定的边界条件、方法和相对步长计算带 NaN 值的雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff_2 = approx_derivative(
            self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
            bounds=(lb, ub))
        jac_diff_3 = approx_derivative(
            self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)

        # 更新初始点
        x0 = 1e-8
        jac_true = self.jac_with_nan(x0)

        # 使用指定的边界条件、方法和相对步长计算带 NaN 值的雅可比矩阵的数值近似，并使用 assert_allclose 断言判断近似值与真实值之间的误差
        jac_diff_2 = approx_derivative(
            self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
            bounds=(lb, ub))
        jac_diff_3 = approx_derivative(
            self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
    def test_non_numpy(self):
        # 设置初始值 x0
        x0 = 1.0
        # 使用非 NumPy 方法计算真实的雅可比矩阵
        jac_true = self.jac_non_numpy(x0)
        # 使用两点法计算数值近似的雅可比矩阵
        jac_diff_2 = approx_derivative(self.jac_non_numpy, x0,
                                       method='2-point')
        # 使用默认方法计算数值近似的雅可比矩阵
        jac_diff_3 = approx_derivative(self.jac_non_numpy, x0)
        # 断言两点法计算的近似雅可比矩阵与真实值之间的接近程度
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        # 断言默认方法计算的近似雅可比矩阵与真实值之间的接近程度
        assert_allclose(jac_diff_3, jac_true, rtol=1e-8)

        # math.exp 不能处理复数参数，因此会引发 TypeError 异常
        assert_raises(TypeError, approx_derivative, self.jac_non_numpy, x0,
                      **dict(method='cs'))

    def test_fp(self):
        # 检查 approx_derivative 在除了64位以外的 FP 大小上的工作情况。
        # 示例源自 gh12991 中的最小工作示例。

        np.random.seed(1)

        def func(p, x):
            return p[0] + p[1] * x

        def err(p, x, y):
            return func(p, x) - y

        # 创建一个包含100个点的浮点数数组 x
        x = np.linspace(0, 1, 100, dtype=np.float64)
        # 创建一个包含100个随机浮点数的数组 y
        y = np.random.random(100).astype(np.float64)
        # 设置初始参数向量 p0
        p0 = np.array([-1.0, -1.0])

        # 使用两点法计算误差函数相对于参数向量 p 的雅可比矩阵（64位浮点数）
        jac_fp64 = approx_derivative(err, p0, method='2-point', args=(x, y))

        # 参数向量 p 是 float32，函数输出是 float64
        jac_fp = approx_derivative(err, p0.astype(np.float32),
                                   method='2-point', args=(x, y))
        # 断言误差函数的输出是 float64 类型
        assert err(p0, x, y).dtype == np.float64
        # 断言不同精度下计算的雅可比矩阵的接近程度
        assert_allclose(jac_fp, jac_fp64, atol=1e-3)

        # 参数向量 p 是 float64，函数输出是 float32
        def err_fp32(p):
            assert p.dtype == np.float32
            return err(p, x, y).astype(np.float32)

        jac_fp = approx_derivative(err_fp32, p0.astype(np.float32),
                                   method='2-point')
        # 断言不同精度下计算的雅可比矩阵的接近程度
        assert_allclose(jac_fp, jac_fp64, atol=1e-3)

        # 检查两点法对导数的误差上限
        def f(x):
            return np.sin(x)

        def g(x):
            return np.cos(x)

        def hess(x):
            return -np.sin(x)

        def calc_atol(h, x0, f, hess, EPS):
            # 截断误差
            t0 = h / 2 * max(np.abs(hess(x0)), np.abs(hess(x0 + h)))
            # 舍入误差
            t1 = EPS / h * max(np.abs(f(x0)), np.abs(f(x0 + h)))
            return t0 + t1

        # 遍历不同的数据类型来计算绝对误差步长 h 和误差上限
        for dtype in [np.float16, np.float32, np.float64]:
            EPS = np.finfo(dtype).eps
            x0 = np.array(1.0).astype(dtype)
            h = _compute_absolute_step(None, x0, f(x0), '2-point')
            atol = calc_atol(h, x0, f, hess, EPS)
            # 计算两点法计算的导数与 g(x0) 的误差，并断言其小于误差上限
            err = approx_derivative(f, x0, method='2-point',
                                    abs_step=h) - g(x0)
            assert abs(err) < atol
    # 定义一个测试函数，用于检查导数的准确性
    def test_check_derivative(self):
        # 设置初始点 x0，并调用 check_derivative 函数计算导数准确性
        x0 = np.array([-10.0, 10])
        accuracy = check_derivative(self.fun_vector_vector,
                                    self.jac_vector_vector, x0)
        # 断言导数的准确性小于 1e-9
        assert_(accuracy < 1e-9)
        
        # 再次调用 check_derivative 函数，检查不同的准确性阈值
        accuracy = check_derivative(self.fun_vector_vector,
                                    self.jac_vector_vector, x0)
        # 断言导数的准确性小于 1e-6
        assert_(accuracy < 1e-6)

        # 设置新的初始点 x0，并调用 check_derivative 函数计算导数准确性
        x0 = np.array([0.0, 0.0])
        accuracy = check_derivative(self.fun_zero_jacobian,
                                    self.jac_zero_jacobian, x0)
        # 断言此时的导数准确性为 0
        assert_(accuracy == 0)
        
        # 再次调用 check_derivative 函数，检查导数的准确性仍然为 0
        accuracy = check_derivative(self.fun_zero_jacobian,
                                    self.jac_zero_jacobian, x0)
        # 断言导数的准确性为 0
        assert_(accuracy == 0)
class TestApproxDerivativeSparse:
    # Example from Numerical Optimization 2nd edition, p. 198.
    def setup_method(self):
        # 设置随机种子，确保可重复性
        np.random.seed(0)
        # 设置测试用例的参数
        self.n = 50
        # 下界
        self.lb = -0.1 * (1 + np.arange(self.n))
        # 上界
        self.ub = 0.1 * (1 + np.arange(self.n))
        # 初始猜测点
        self.x0 = np.empty(self.n)
        self.x0[::2] = (1 - 1e-7) * self.lb[::2]
        self.x0[1::2] = (1 - 1e-7) * self.ub[1::2]

        # 真实的雅可比矩阵
        self.J_true = self.jac(self.x0)

    # 定义目标函数
    def fun(self, x):
        e = x[1:]**3 - x[:-1]**2
        return np.hstack((0, 3 * e)) + np.hstack((2 * e, 0))

    # 计算雅可比矩阵
    def jac(self, x):
        n = x.size
        J = np.zeros((n, n))
        J[0, 0] = -4 * x[0]
        J[0, 1] = 6 * x[1]**2
        for i in range(1, n - 1):
            J[i, i - 1] = -6 * x[i-1]
            J[i, i] = 9 * x[i]**2 - 4 * x[i]
            J[i, i + 1] = 6 * x[i+1]**2
        J[-1, -1] = 9 * x[-1]**2
        J[-1, -2] = -6 * x[-2]

        return J

    # 构造稀疏结构矩阵
    def structure(self, n):
        A = np.zeros((n, n), dtype=int)
        A[0, 0] = 1
        A[0, 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1: i + 2] = 1
        A[-1, -1] = 1
        A[-1, -2] = 1

        return A

    # 测试所有情况
    def test_all(self):
        # 获取结构矩阵
        A = self.structure(self.n)
        # 初始顺序
        order = np.arange(self.n)
        # 对顺序进行随机打乱
        np.random.shuffle(order)
        # 获取分组信息
        groups_1 = group_columns(A, order)
        np.random.shuffle(order)
        groups_2 = group_columns(A, order)

        # 对于不同的求导方法、分组信息、下界和上界进行组合测试
        for method, groups, l, u in product(
                ['2-point', '3-point', 'cs'], [groups_1, groups_2],
                [-np.inf, self.lb], [np.inf, self.ub]):
            # 计算近似导数
            J = approx_derivative(self.fun, self.x0, method=method,
                                  bounds=(l, u), sparsity=(A, groups))
            # 断言返回结果是 CSR 稀疏矩阵
            assert_(isinstance(J, csr_matrix))
            # 断言近似导数的数组表示与真实雅可比矩阵的接近程度
            assert_allclose(J.toarray(), self.J_true, rtol=1e-6)

            # 设置相对步长
            rel_step = np.full_like(self.x0, 1e-8)
            rel_step[::2] *= -1
            # 重新计算近似导数
            J = approx_derivative(self.fun, self.x0, method=method,
                                  rel_step=rel_step, sparsity=(A, groups))
            # 断言近似导数的数组表示与真实雅可比矩阵的接近程度
            assert_allclose(J.toarray(), self.J_true, rtol=1e-5)

    # 测试没有预计算分组的情况
    def test_no_precomputed_groups(self):
        # 获取结构矩阵
        A = self.structure(self.n)
        # 计算近似导数
        J = approx_derivative(self.fun, self.x0, sparsity=A)
        # 断言近似导数的数组表示与真实雅可比矩阵的接近程度
        assert_allclose(J.toarray(), self.J_true, rtol=1e-6)

    # 测试等价性
    def test_equivalence(self):
        # 构造全为1的结构矩阵和顺序分组
        structure = np.ones((self.n, self.n), dtype=int)
        groups = np.arange(self.n)
        # 对于不同的求导方法进行测试
        for method in ['2-point', '3-point', 'cs']:
            # 计算稠密雅可比矩阵
            J_dense = approx_derivative(self.fun, self.x0, method=method)
            # 计算稀疏雅可比矩阵
            J_sparse = approx_derivative(
                self.fun, self.x0, sparsity=(structure, groups), method=method)
            # 断言稠密和稀疏雅可比矩阵的接近程度
            assert_allclose(J_dense, J_sparse.toarray(),
                            rtol=5e-16, atol=7e-15)
    # 定义内部函数 jac，用于计算雅可比矩阵
    def jac(x):
        return csr_matrix(self.jac(x))

    # 使用 check_derivative 函数检查 self.fun 函数与 jac 函数的导数精度
    accuracy = check_derivative(self.fun, jac, self.x0,
                                bounds=(self.lb, self.ub))
    # 断言精度小于 1e-9，即导数的数值精度应足够高
    assert_(accuracy < 1e-9)

    # 再次使用 check_derivative 函数进行导数检查
    accuracy = check_derivative(self.fun, jac, self.x0,
                                bounds=(self.lb, self.ub))
    # 断言精度小于 1e-9，确保导数的数值精度符合预期
    assert_(accuracy < 1e-9)
# 定义一个测试类 TestApproxDerivativeLinearOperator，用于测试近似导数线性操作符的准确性
class TestApproxDerivativeLinearOperator:

    # 定义一个计算单变量标量函数 np.sinh(x) 的方法
    def fun_scalar_scalar(self, x):
        return np.sinh(x)

    # 定义一个计算单变量标量函数 np.cosh(x) 的方法
    def jac_scalar_scalar(self, x):
        return np.cosh(x)

    # 定义一个计算单变量向量函数 np.array([x[0]**2, np.tan(x[0]), np.exp(x[0])]) 的方法
    def fun_scalar_vector(self, x):
        return np.array([x[0]**2, np.tan(x[0]), np.exp(x[0])])

    # 定义一个计算单变量向量函数的雅可比矩阵 np.array([2 * x[0], np.cos(x[0]) ** -2, np.exp(x[0])]).reshape(-1, 1) 的方法
    def jac_scalar_vector(self, x):
        return np.array([2 * x[0], np.cos(x[0]) ** -2, np.exp(x[0])]).reshape(-1, 1)

    # 定义一个计算向量变量函数 np.sin(x[0] * x[1]) * np.log(x[0]) 的方法
    def fun_vector_scalar(self, x):
        return np.sin(x[0] * x[1]) * np.log(x[0])

    # 定义一个计算向量变量函数的雅可比向量 np.array([x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) + np.sin(x[0] * x[1]) / x[0],
    #            x[0] * np.cos(x[0] * x[1]) * np.log(x[0])]) 的方法
    def jac_vector_scalar(self, x):
        return np.array([
            x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) +
            np.sin(x[0] * x[1]) / x[0],
            x[0] * np.cos(x[0] * x[1]) * np.log(x[0])
        ])

    # 定义一个计算向量变量函数 np.array([x[0] * np.sin(x[1]), x[1] * np.cos(x[0]), x[0] ** 3 * x[1] ** -0.5]) 的方法
    def fun_vector_vector(self, x):
        return np.array([
            x[0] * np.sin(x[1]),
            x[1] * np.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    # 定义一个计算向量变量函数的雅可比矩阵 np.array([[np.sin(x[1]), x[0] * np.cos(x[1])],
    #            [-x[1] * np.sin(x[0]), np.cos(x[0])],
    #            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]]) 的方法
    def jac_vector_vector(self, x):
        return np.array([
            [np.sin(x[1]), x[0] * np.cos(x[1])],
            [-x[1] * np.sin(x[0]), np.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    # 定义一个测试方法 test_scalar_scalar，用于测试近似导数操作符在单变量标量函数上的准确性
    def test_scalar_scalar(self):
        x0 = 1.0
        # 使用近似导数函数 approx_derivative 计算 fun_scalar_scalar 在 x0 处的导数，使用两点方法
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        # 使用近似导数函数 approx_derivative 计算 fun_scalar_scalar 在 x0 处的导数，使用默认方法
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0,
                                       as_linear_operator=True)
        # 使用近似导数函数 approx_derivative 计算 fun_scalar_scalar 在 x0 处的导数，使用中心差分方法
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        # 计算 fun_scalar_scalar 在 x0 处的真实导数
        jac_true = self.jac_scalar_scalar(x0)
        np.random.seed(1)
        # 对随机生成的向量 p 进行多次断言，验证近似导数的准确性
        for i in range(10):
            p = np.random.uniform(-10, 10, size=(1,))
            assert_allclose(jac_diff_2.dot(p), jac_true*p,
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true*p,
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true*p,
                            rtol=5e-6)
    # 测试标量到向量函数的数值梯度计算
    def test_scalar_vector(self):
        # 设置初始点 x0
        x0 = 0.5
        # 使用数值逼近计算梯度，使用两点公式
        jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        # 使用数值逼近计算梯度，默认使用三点公式
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0,
                                       as_linear_operator=True)
        # 使用复数步长法数值逼近计算梯度
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        # 计算真实梯度
        jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
        # 设置随机种子
        np.random.seed(1)
        # 进行10次随机测试
        for i in range(10):
            # 生成随机向量 p
            p = np.random.uniform(-10, 10, size=(1,))
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p),
                            rtol=1e-5)
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p),
                            rtol=5e-6)
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p),
                            rtol=5e-6)

    # 测试向量到标量函数的数值梯度计算
    def test_vector_scalar(self):
        # 设置初始点 x0
        x0 = np.array([100.0, -0.5])
        # 使用数值逼近计算梯度，使用两点公式
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        # 使用数值逼近计算梯度，默认使用三点公式
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0,
                                       as_linear_operator=True)
        # 使用复数步长法数值逼近计算梯度
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        # 计算真实梯度
        jac_true = self.jac_vector_scalar(x0)
        # 设置随机种子
        np.random.seed(1)
        # 进行10次随机测试
        for i in range(10):
            # 生成随机向量 p
            p = np.random.uniform(-10, 10, size=x0.shape)
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_2.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=1e-5)
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_3.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=5e-6)
            # 检查数值逼近梯度和真实梯度的点积是否接近
            assert_allclose(jac_diff_4.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=1e-7)
    # 定义一个测试方法，用于测试向量到向量函数的数值导数计算
    def test_vector_vector(self):
        # 创建一个包含两个元素的 NumPy 数组，作为测试输入向量 x0
        x0 = np.array([-100.0, 0.2])
        # 使用数值方法 '2-point' 计算函数 self.fun_vector_vector 在 x0 处的数值导数，并返回线性操作符
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        # 使用默认的数值方法计算函数 self.fun_vector_vector 在 x0 处的数值导数，并返回线性操作符
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
                                       as_linear_operator=True)
        # 使用复数步长数值方法 'cs' 计算函数 self.fun_vector_vector 在 x0 处的数值导数，并返回线性操作符
        jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        # 计算函数 self.jac_vector_vector 在 x0 处的精确雅可比矩阵
        jac_true = self.jac_vector_vector(x0)
        # 设定随机数种子为1，生成 10 组在 [-10, 10] 范围内均匀分布的与 x0 形状相同的向量 p
        np.random.seed(1)
        for i in range(10):
            # 取出当前随机生成的向量 p
            p = np.random.uniform(-10, 10, size=x0.shape)
            # 断言：使用 '2-point' 方法计算的数值雅可比矩阵与真实雅可比矩阵在 p 上的乘积的近似相等
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p), rtol=1e-5)
            # 断言：使用默认方法计算的数值雅可比矩阵与真实雅可比矩阵在 p 上的乘积的近似相等
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p), rtol=1e-6)
            # 断言：使用 'cs' 方法计算的数值雅可比矩阵与真实雅可比矩阵在 p 上的乘积的近似相等
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p), rtol=1e-7)

    # 定义一个测试异常情况的方法
    def test_exception(self):
        # 创建一个包含两个元素的 NumPy 数组，作为测试输入向量 x0
        x0 = np.array([-100.0, 0.2])
        # 断言：调用 approx_derivative 函数时，传入方法 '2-point' 和超出界限 (1, np.inf) 的参数会引发 ValueError 异常
        assert_raises(ValueError, approx_derivative,
                      self.fun_vector_vector, x0,
                      method='2-point', bounds=(1, np.inf))
def test_absolute_step_sign():
    # test for gh12487
    # 如果对于二点差分指定了绝对步长，请确保方向对应于步长。
    # 即，如果步长为正，则应使用前向差分；如果步长为负，则应使用后向差分。

    # 在 x = [-1, -1] 处有双重不连续点
    # 第一个分量为 \/, 第二个分量为 /\
    def f(x):
        return -np.abs(x[0] + 1) + np.abs(x[1] + 1)

    # 检查使用前向差分
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=1e-8)
    assert_allclose(grad, [-1.0, 1.0])

    # 检查使用后向差分
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=-1e-8)
    assert_allclose(grad, [1.0, -1.0])

    # 检查两个参数都使用步长的前向差分
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[1e-8, 1e-8]
    )
    assert_allclose(grad, [-1.0, 1.0])

    # 检查可以混合使用前向和后向步长
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[1e-8, -1e-8]
     )
    assert_allclose(grad, [-1.0, -1.0])
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[-1e-8, 1e-8]
    )
    assert_allclose(grad, [1.0, 1.0])

    # 如果步长遇到边界，前向步长应该变成后向步长
    # 这在 TestAdjustSchemeToBounds 中有测试，但只针对更低级的函数。
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=1e-8,
        bounds=(-np.inf, -1)
    )
    assert_allclose(grad, [1.0, -1.0])

    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=-1e-8, bounds=(-1, np.inf)
    )
    assert_allclose(grad, [-1.0, 1.0])


def test__compute_absolute_step():
    # 测试从相对步长计算绝对步长的函数
    methods = ['2-point', '3-point', 'cs']

    x0 = np.array([1e-5, 0, 1, 1e5])

    EPS = np.finfo(np.float64).eps
    relative_step = {
        "2-point": EPS**0.5,
        "3-point": EPS**(1/3),
        "cs": EPS**0.5
    }
    f0 = np.array(1.0)

    for method in methods:
        rel_step = relative_step[method]
        correct_step = np.array([rel_step,
                                 rel_step * 1.,
                                 rel_step * 1.,
                                 rel_step * np.abs(x0[3])])

        abs_step = _compute_absolute_step(None, x0, f0, method)
        assert_allclose(abs_step, correct_step)

        sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
        abs_step = _compute_absolute_step(None, -x0, f0, method)
        assert_allclose(abs_step, sign_x0 * correct_step)

    # 如果提供了相对步长，则应使用它
    rel_step = np.array([0.1, 1, 10, 100])
    # 计算正确步长，使用 numpy 数组表示，包括 x0 中的第一个元素与 rel_step 中的第一个元素的乘积，
    # '2-point' 键对应的 relative_step 的值，rel_step 中索引为 2 和 3 的元素乘以 1 和 x0 中第四个元素的绝对值。
    correct_step = np.array([rel_step[0] * x0[0],
                             relative_step['2-point'],
                             rel_step[2] * 1.,
                             rel_step[3] * np.abs(x0[3])])
    
    # 计算绝对步长，调用 _compute_absolute_step 函数，传入 rel_step, -x0, f0 和 '2-point' 作为参数。
    abs_step = _compute_absolute_step(rel_step, x0, f0, '2-point')
    # 使用 assert_allclose 函数断言 abs_step 应与 correct_step 很接近，即它们的值应该几乎相等。
    assert_allclose(abs_step, correct_step)
    
    # 计算 x0 中每个元素的符号，使用 (-x0 >= 0).astype(float) * 2 - 1 的方式，将 x0 中每个元素是否大于等于零的结果转换为浮点数后乘以 2 再减 1。
    sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
    # 计算绝对步长，调用 _compute_absolute_step 函数，传入 rel_step, -x0, f0 和 '2-point' 作为参数。
    abs_step = _compute_absolute_step(rel_step, -x0, f0, '2-point')
    # 使用 assert_allclose 函数断言 abs_step 应与 sign_x0 乘以 correct_step 的结果很接近。
    assert_allclose(abs_step, sign_x0 * correct_step)
```