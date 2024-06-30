# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_lsq_common.py`

```
# 从 numpy.testing 模块导入断言函数 assert_, assert_allclose, assert_equal
# 从 pytest 模块导入 raises 并起别名为 assert_raises
import numpy as np

# 从 scipy.optimize._lsq.common 模块导入以下函数
from scipy.optimize._lsq.common import (
    step_size_to_bound,              # 导入 step_size_to_bound 函数
    find_active_constraints,         # 导入 find_active_constraints 函数
    make_strictly_feasible,         # 导入 make_strictly_feasible 函数
    CL_scaling_vector,               # 导入 CL_scaling_vector 函数
    intersect_trust_region,          # 导入 intersect_trust_region 函数
    build_quadratic_1d,              # 导入 build_quadratic_1d 函数
    minimize_quadratic_1d,           # 导入 minimize_quadratic_1d 函数
    evaluate_quadratic,              # 导入 evaluate_quadratic 函数
    reflective_transformation,       # 导入 reflective_transformation 函数
    left_multiplied_operator,        # 导入 left_multiplied_operator 函数
    right_multiplied_operator        # 导入 right_multiplied_operator 函数
)


class TestBounds:
    # 定义 TestBounds 类，用于测试边界函数

    def test_step_size_to_bounds(self):
        # 测试 step_size_to_bound 函数

        lb = np.array([-1.0, 2.5, 10.0])
        # 定义下界 lb

        ub = np.array([1.0, 5.0, 100.0])
        # 定义上界 ub

        x = np.array([0.0, 2.5, 12.0])
        # 定义输入向量 x

        s = np.array([0.1, 0.0, 0.0])
        # 定义步长向量 s

        step, hits = step_size_to_bound(x, s, lb, ub)
        # 调用 step_size_to_bound 函数计算步长和碰撞信息

        assert_equal(step, 10)
        # 使用 assert_equal 断言步长结果为 10

        assert_equal(hits, [1, 0, 0])
        # 使用 assert_equal 断言碰撞信息列表为 [1, 0, 0]

        s = np.array([0.01, 0.05, -1.0])
        # 更新步长向量 s

        step, hits = step_size_to_bound(x, s, lb, ub)
        # 重新计算步长和碰撞信息

        assert_equal(step, 2)
        # 使用 assert_equal 断言步长结果为 2

        assert_equal(hits, [0, 0, -1])
        # 使用 assert_equal 断言碰撞信息列表为 [0, 0, -1]

        s = np.array([10.0, -0.0001, 100.0])
        # 更新步长向量 s

        step, hits = step_size_to_bound(x, s, lb, ub)
        # 重新计算步长和碰撞信息

        assert_equal(step, np.array(-0))
        # 使用 assert_equal 断言步长结果为 -0

        assert_equal(hits, [0, -1, 0])
        # 使用 assert_equal 断言碰撞信息列表为 [0, -1, 0]

        s = np.array([1.0, 0.5, -2.0])
        # 更新步长向量 s

        step, hits = step_size_to_bound(x, s, lb, ub)
        # 重新计算步长和碰撞信息

        assert_equal(step, 1.0)
        # 使用 assert_equal 断言步长结果为 1.0

        assert_equal(hits, [1, 0, -1])
        # 使用 assert_equal 断言碰撞信息列表为 [1, 0, -1]

        s = np.zeros(3)
        # 更新步长向量 s

        step, hits = step_size_to_bound(x, s, lb, ub)
        # 重新计算步长和碰撞信息

        assert_equal(step, np.inf)
        # 使用 assert_equal 断言步长结果为正无穷

        assert_equal(hits, [0, 0, 0])
        # 使用 assert_equal 断言碰撞信息列表为 [0, 0, 0]

    def test_find_active_constraints(self):
        # 测试 find_active_constraints 函数

        lb = np.array([0.0, -10.0, 1.0])
        # 定义下界 lb

        ub = np.array([1.0, 0.0, 100.0])
        # 定义上界 ub

        x = np.array([0.5, -5.0, 2.0])
        # 定义输入向量 x

        active = find_active_constraints(x, lb, ub)
        # 调用 find_active_constraints 函数计算活动约束

        assert_equal(active, [0, 0, 0])
        # 使用 assert_equal 断言活动约束结果为 [0, 0, 0]

        x = np.array([0.0, 0.0, 10.0])
        # 更新输入向量 x

        active = find_active_constraints(x, lb, ub)
        # 重新计算活动约束

        assert_equal(active, [-1, 1, 0])
        # 使用 assert_equal 断言活动约束结果为 [-1, 1, 0]

        active = find_active_constraints(x, lb, ub, rtol=0)
        # 带有相对容差的重新计算活动约束

        assert_equal(active, [-1, 1, 0])
        # 使用 assert_equal 断言活动约束结果为 [-1, 1, 0]

        x = np.array([1e-9, -1e-8, 100 - 1e-9])
        # 更新输入向量 x

        active = find_active_constraints(x, lb, ub)
        # 重新计算活动约束

        assert_equal(active, [0, 0, 1])
        # 使用 assert_equal 断言活动约束结果为 [0, 0, 1]

        active = find_active_constraints(x, lb, ub, rtol=1.5e-9)
        # 带有相对容差的重新计算活动约束

        assert_equal(active, [-1, 0, 1])
        # 使用 assert_equal 断言活动约束结果为 [-1, 0, 1]

        lb = np.array([1.0, -np.inf, -np.inf])
        # 更新下界 lb

        ub = np.array([np.inf, 10.0, np.inf])
        # 更新上界 ub

        x = np.ones(3)
        # 更新输入向量 x

        active = find_active_constraints(x, lb, ub)
        # 重新计算活动约束

        assert_equal(active, [-1, 0, 0])
        # 使用 assert_equal 断言活动约束结果为 [-1, 0, 0]

        x = np.array([0.0, 11.0, 0.0])
        # 更新输入向量 x

        active = find_active_constraints(x, lb, ub)
        # 重新计算活动约束

        assert_equal(active, [-1, 1, 0])
        # 使用 assert_equal 断言活动约束结果为 [-1, 1, 0]

        active = find_active_constraints(x, lb, ub, rtol=0)
        # 带有相对容差的重新计算活动约束

        assert_equal(active, [-1, 1, 0])
        # 使用 assert_equal 断言活动约束结果为 [-1, 1, 0]
    # 定义一个测试函数，用于测试 make_strictly_feasible 函数的行为
    def test_make_strictly_feasible(self):
        # 创建下界 lb 和上界 ub 的 numpy 数组
        lb = np.array([-0.5, -0.8, 2.0])
        ub = np.array([0.8, 1.0, 3.0])

        # 创建一个测试输入向量 x，其中第三个元素略大于 2.0
        x = np.array([-0.5, 0.0, 2 + 1e-10])

        # 调用 make_strictly_feasible 函数，将 x 调整为严格可行解，rstep 参数为 0
        x_new = make_strictly_feasible(x, lb, ub, rstep=0)
        # 断言结果的第一个元素大于 -0.5
        assert_(x_new[0] > -0.5)
        # 断言结果的其余元素与输入 x 的对应位置元素相同
        assert_equal(x_new[1:], x[1:])

        # 再次调用 make_strictly_feasible 函数，此时 rstep 参数为 1e-4
        x_new = make_strictly_feasible(x, lb, ub, rstep=1e-4)
        # 断言结果与预期的向量匹配
        assert_equal(x_new, [-0.5 + 1e-4, 0.0, 2 * (1 + 1e-4)])

        # 更换测试输入向量 x，其中包含一个超出上界的值和一个超出下界的值
        x = np.array([-0.5, -1, 3.1])
        # 调用 make_strictly_feasible 函数，将 x 调整为严格可行解，默认 rstep
        x_new = make_strictly_feasible(x, lb, ub)
        # 断言所有结果元素均在对应的 lb 和 ub 范围内
        assert_(np.all((x_new >= lb) & (x_new <= ub)))

        # 再次调用 make_strictly_feasible 函数，此时 rstep 参数为 0
        x_new = make_strictly_feasible(x, lb, ub, rstep=0)
        # 断言所有结果元素均在对应的 lb 和 ub 范围内
        assert_(np.all((x_new >= lb) & (x_new <= ub)))

        # 创建新的下界 lb 和上界 ub 的 numpy 数组
        lb = np.array([-1, 100.0])
        ub = np.array([1, 100.0 + 1e-10])
        # 创建一个测试输入向量 x，包含边界上的值
        x = np.array([0, 100.0])
        # 调用 make_strictly_feasible 函数，将 x 调整为严格可行解，rstep 参数为 1e-8
        x_new = make_strictly_feasible(x, lb, ub, rstep=1e-8)
        # 断言结果与预期的向量匹配
        assert_equal(x_new, [0, 100.0 + 0.5e-10])

    # 定义一个测试函数，用于测试 CL_scaling_vector 函数的行为
    def test_scaling_vector(self):
        # 创建下界 lb 和上界 ub 的 numpy 数组，以及输入向量 x 和梯度向量 g
        lb = np.array([-np.inf, -5.0, 1.0, -np.inf])
        ub = np.array([1.0, np.inf, 10.0, np.inf])
        x = np.array([0.5, 2.0, 5.0, 0.0])
        g = np.array([1.0, 0.1, -10.0, 0.0])

        # 调用 CL_scaling_vector 函数，获取结果向量 v 和 dv
        v, dv = CL_scaling_vector(x, g, lb, ub)
        # 断言结果向量 v 与预期的向量匹配
        assert_equal(v, [1.0, 7.0, 5.0, 1.0])
        # 断言导数向量 dv 与预期的向量匹配
        assert_equal(dv, [0.0, 1.0, -1.0, 0.0])
class TestQuadraticFunction:
    # 设置测试方法的初始化
    def setup_method(self):
        # 初始化 J 矩阵，包含三个行向量
        self.J = np.array([
            [0.1, 0.2],
            [-1.0, 1.0],
            [0.5, 0.2]])
        # 初始化 g 向量
        self.g = np.array([0.8, -2.0])
        # 初始化 diag 向量
        self.diag = np.array([1.0, 2.0])

    # 测试构建一维二次函数
    def test_build_quadratic_1d(self):
        # 初始化长度为 2 的零向量 s
        s = np.zeros(2)
        # 调用 build_quadratic_1d 函数，期望返回结果 a 和 b 均为 0
        a, b = build_quadratic_1d(self.J, self.g, s)
        assert_equal(a, 0)
        assert_equal(b, 0)

        # 传入 diag 向量，再次调用 build_quadratic_1d 函数，期望返回结果 a 和 b 均为 0
        a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
        assert_equal(a, 0)
        assert_equal(b, 0)

        # 修改 s 向量内容
        s = np.array([1.0, -1.0])
        # 调用 build_quadratic_1d 函数，期望返回结果 a 为 2.05，b 为 2.8
        a, b = build_quadratic_1d(self.J, self.g, s)
        assert_equal(a, 2.05)
        assert_equal(b, 2.8)

        # 传入 diag 向量，再次调用 build_quadratic_1d 函数，期望返回结果 a 为 3.55，b 为 2.8
        a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
        assert_equal(a, 3.55)
        assert_equal(b, 2.8)

        # 初始化 s0 向量
        s0 = np.array([0.5, 0.5])
        # 传入 diag 向量和 s0 向量，再次调用 build_quadratic_1d 函数
        # 期望返回结果 a 为 3.55，b 接近 2.39，c 接近 -0.1525
        a, b, c = build_quadratic_1d(self.J, self.g, s, diag=self.diag, s0=s0)
        assert_equal(a, 3.55)
        assert_allclose(b, 2.39)
        assert_allclose(c, -0.1525)

    # 测试最小化一维二次函数
    def test_minimize_quadratic_1d(self):
        # 初始化 a 和 b
        a = 5
        b = -1

        # 调用 minimize_quadratic_1d 函数，期望返回结果 t 为 1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, 1, 2)
        assert_equal(t, 1)
        assert_allclose(y, a * t**2 + b * t, rtol=1e-15)

        # 调用 minimize_quadratic_1d 函数，期望返回结果 t 为 -1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, -2, -1)
        assert_equal(t, -1)
        assert_allclose(y, a * t**2 + b * t, rtol=1e-15)

        # 调用 minimize_quadratic_1d 函数，期望返回结果 t 为 0.1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, -1, 1)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t**2 + b * t, rtol=1e-15)

        # 初始化 c
        c = 10
        # 传入额外参数 c，调用 minimize_quadratic_1d 函数，期望返回结果 t 为 0.1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, -1, 1, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t**2 + b * t + c, rtol=1e-15)

        # 传入边界 -np.inf 和 np.inf，调用 minimize_quadratic_1d 函数，期望返回结果 t 为 0.1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)

        # 传入边界 0 和 np.inf，调用 minimize_quadratic_1d 函数，期望返回结果 t 为 0.1，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, 0, np.inf, c=c)
        assert_equal(t, 0.1)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)

        # 传入边界 -np.inf 和 0，调用 minimize_quadratic_1d 函数，期望返回结果 t 为 0，y 满足二次函数形式
        t, y = minimize_quadratic_1d(a, b, -np.inf, 0, c=c)
        assert_equal(t, 0)
        assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)

        # 修改 a 和 b 的值
        a = -1
        b = 0.2
        # 调用 minimize_quadratic_1d 函数，期望返回结果 y 为 -np.inf
        t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf)
        assert_equal(y, -np.inf)

        # 调用 minimize_quadratic_1d 函数，期望返回结果 t 为 np.inf，y 为 -np.inf
        t, y = minimize_quadratic_1d(a, b, 0, np.inf)
        assert_equal(t, np.inf)
        assert_equal(y, -np.inf)

        # 调用 minimize_quadratic_1d 函数，期望返回结果 t 为 -np.inf，y 为 -np.inf
        t, y = minimize_quadratic_1d(a, b, -np.inf, 0)
        assert_equal(t, -np.inf)
        assert_equal(y, -np.inf)
    # 定义一个测试方法，用于测试 evaluate_quadratic 函数的不同情况

    # 创建一个包含两个元素的 NumPy 数组 s，表示单一维度的情况
    s = np.array([1.0, -1.0])

    # 调用 evaluate_quadratic 函数计算二次方程的值，使用默认的 diag 参数
    value = evaluate_quadratic(self.J, self.g, s)
    # 断言计算结果与预期结果 4.85 相等
    assert_equal(value, 4.85)

    # 再次调用 evaluate_quadratic 函数，这次使用传入的 diag 参数
    value = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
    # 断言计算结果与预期结果 6.35 相等
    assert_equal(value, 6.35)

    # 创建一个包含三个行两个列的 NumPy 数组 s，表示多维度的情况
    s = np.array([[1.0, -1.0],
                 [1.0, 1.0],
                 [0.0, 0.0]])

    # 调用 evaluate_quadratic 函数计算多个二次方程的值，使用默认的 diag 参数
    values = evaluate_quadratic(self.J, self.g, s)
    # 断言计算结果与预期结果列表 [4.85, -0.91, 0.0] 接近
    assert_allclose(values, [4.85, -0.91, 0.0])

    # 再次调用 evaluate_quadratic 函数，这次使用传入的 diag 参数
    values = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
    # 断言计算结果与预期结果列表 [6.35, 0.59, 0.0] 接近
    assert_allclose(values, [6.35, 0.59, 0.0])
class TestTrustRegion:
    # 定义测试类 TestTrustRegion
    def test_intersect(self):
        # 定义测试方法 test_intersect
        Delta = 1.0
        # 设置 Delta 值为 1.0

        x = np.zeros(3)
        # 初始化 x 为一个长度为 3 的零向量
        s = np.array([1.0, 0.0, 0.0])
        # 初始化 s 为 [1.0, 0.0, 0.0]
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        # 调用 intersect_trust_region 函数计算给定 x, s, Delta 下的 t_neg 和 t_pos
        assert_equal(t_neg, -1)
        # 断言 t_neg 等于 -1
        assert_equal(t_pos, 1)
        # 断言 t_pos 等于 1

        s = np.array([-1.0, 1.0, -1.0])
        # 更新 s 为 [-1.0, 1.0, -1.0]
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        # 再次调用 intersect_trust_region 函数计算 t_neg 和 t_pos
        assert_allclose(t_neg, -3**-0.5)
        # 使用 assert_allclose 断言 t_neg 接近于 -sqrt(3)
        assert_allclose(t_pos, 3**-0.5)
        # 使用 assert_allclose 断言 t_pos 接近于 sqrt(3)

        x = np.array([0.5, -0.5, 0])
        # 更新 x 为 [0.5, -0.5, 0]
        s = np.array([0, 0, 1.0])
        # 更新 s 为 [0, 0, 1.0]
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        # 再次调用 intersect_trust_region 函数计算 t_neg 和 t_pos
        assert_allclose(t_neg, -2**-0.5)
        # 使用 assert_allclose 断言 t_neg 接近于 -sqrt(2)
        assert_allclose(t_pos, 2**-0.5)
        # 使用 assert_allclose 断言 t_pos 接近于 sqrt(2)

        x = np.ones(3)
        # 更新 x 为一个长度为 3 的全一向量
        assert_raises(ValueError, intersect_trust_region, x, s, Delta)
        # 使用 assert_raises 断言在给定 x, s, Delta 的情况下会引发 ValueError 异常

        x = np.zeros(3)
        s = np.zeros(3)
        # 将 x 和 s 都设置为长度为 3 的零向量
        assert_raises(ValueError, intersect_trust_region, x, s, Delta)
        # 使用 assert_raises 断言在给定 x, s, Delta 的情况下会引发 ValueError 异常


def test_reflective_transformation():
    # 定义测试函数 test_reflective_transformation
    lb = np.array([-1, -2], dtype=float)
    # 初始化 lb 为 [-1, -2]，数据类型为 float
    ub = np.array([5, 3], dtype=float)
    # 初始化 ub 为 [5, 3]，数据类型为 float

    y = np.array([0, 0])
    # 初始化 y 为 [0, 0]
    x, g = reflective_transformation(y, lb, ub)
    # 调用 reflective_transformation 函数计算给定 y, lb, ub 下的 x 和 g
    assert_equal(x, y)
    # 断言 x 等于 y
    assert_equal(g, np.ones(2))
    # 断言 g 等于长度为 2 的全一向量

    y = np.array([-4, 4], dtype=float)
    # 更新 y 为 [-4, 4]
    x, g = reflective_transformation(y, lb, np.array([np.inf, np.inf]))
    # 再次调用 reflective_transformation 函数计算 x 和 g
    assert_equal(x, [2, 4])
    # 断言 x 等于 [2, 4]
    assert_equal(g, [-1, 1])
    # 断言 g 等于 [-1, 1]

    x, g = reflective_transformation(y, np.array([-np.inf, -np.inf]), ub)
    # 再次调用 reflective_transformation 函数计算 x 和 g
    assert_equal(x, [-4, 2])
    # 断言 x 等于 [-4, 2]
    assert_equal(g, [1, -1])
    # 断言 g 等于 [1, -1]

    x, g = reflective_transformation(y, lb, ub)
    # 再次调用 reflective_transformation 函数计算 x 和 g
    assert_equal(x, [2, 2])
    # 断言 x 等于 [2, 2]
    assert_equal(g, [-1, -1])
    # 断言 g 等于 [-1, -1]

    lb = np.array([-np.inf, -2])
    # 更新 lb 为 [-inf, -2]
    ub = np.array([5, np.inf])
    # 更新 ub 为 [5, inf]
    y = np.array([10, 10], dtype=float)
    # 初始化 y 为 [10, 10]
    x, g = reflective_transformation(y, lb, ub)
    # 再次调用 reflective_transformation 函数计算 x 和 g
    assert_equal(x, [0, 10])
    # 断言 x 等于 [0, 10]
    assert_equal(g, [-1, 1])
    # 断言 g 等于 [-1, 1]


def test_linear_operators():
    # 定义测试函数 test_linear_operators
    A = np.arange(6).reshape((3, 2))
    # 初始化 A 为 reshape 后的 arange(6)，形状为 (3, 2)

    d_left = np.array([-1, 2, 5])
    # 初始化 d_left 为 [-1, 2, 5]
    DA = np.diag(d_left).dot(A)
    # 计算对角矩阵 np.diag(d_left) 与 A 的乘积，赋值给 DA
    J_left = left_multiplied_operator(A, d_left)
    # 调用 left_multiplied_operator 函数生成以 A 和 d_left 为参数的 J_left

    d_right = np.array([5, 10])
    # 初始化 d_right 为 [5, 10]
    AD = A.dot(np.diag(d_right))
    # 计算 A 与对角矩阵 np.diag(d_right) 的乘积，赋值给 AD
    J_right = right_multiplied_operator(A, d_right)
    # 调用 right_multiplied_operator 函数生成以 A 和 d_right 为参数的 J_right

    x = np.array([-2, 3])
    # 初始化 x 为 [-2, 3]
    X = -2 * np.arange(2, 8).reshape((2, 3))
    # 初始化 X 为 reshape 后的 -2 * arange(2, 8)，形状为 (2, 3)
    xt = np.array([0, -2, 15])
    # 初始化 xt 为 [0, -2, 15]

    assert_allclose(DA.dot(x), J_left.matvec(x))
    # 使用 assert_allclose 断言 DA 与 x 的点积接近于 J_left 对 x 的 matvec 操作结果
    assert_allclose(DA.dot(X), J_left.matmat(X))
    # 使用 assert_allclose 断言 DA 与 X 的矩阵乘积接近于 J_left 对 X 的 matmat 操作结果
    assert_allclose(DA.T.dot(xt), J_left.rmatvec(xt))
    # 使用 assert_allclose 断言 DA 的转置与 xt 的点积接近于 J_left 对 xt 的 rmatvec 操作结果

    assert_allclose(AD.dot(x), J_right.matvec(x))
    # 使用 assert_allclose 断言 AD 与 x 的点积接近于 J_right 对 x 的 matvec 操作结果
    assert_allclose(AD.dot(X), J_right.matmat(X))
    # 使用 assert_allclose 断言 AD 与 X 的矩阵乘积接近于 J_right 对 X 的 matmat 操作结果
    assert_allclose(AD.T.dot(xt), J_right.rmatvec(xt))
    # 使用 assert_allclose 断言 AD 的转置与 xt 的点积接近于 J_right 对 xt 的 rmatvec 操作结果
```