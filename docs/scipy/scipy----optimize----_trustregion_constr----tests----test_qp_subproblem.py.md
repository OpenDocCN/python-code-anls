# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tests\test_qp_subproblem.py`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np
# 从 SciPy 稀疏矩阵模块中导入 csc_matrix 函数
from scipy.sparse import csc_matrix
# 从 scipy.optimize._trustregion_constr.qp_subproblem 模块中导入多个函数
# 这些函数包括 eqp_kktfact, projected_cg, box_intersections,
# sphere_intersections, box_sphere_intersections, modified_dogleg
from scipy.optimize._trustregion_constr.qp_subproblem \
    import (eqp_kktfact,
            projected_cg,
            box_intersections,
            sphere_intersections,
            box_sphere_intersections,
            modified_dogleg)
# 从 scipy.optimize._trustregion_constr.projections 模块中导入 projections 函数
from scipy.optimize._trustregion_constr.projections \
    import projections
# 从 NumPy 测试模块中导入 TestCase 类、assert_array_almost_equal 和 assert_equal 函数
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
# 导入 pytest 库
import pytest


# 定义一个测试类 TestEQPDirectFactorization，继承自 TestCase 类
class TestEQPDirectFactorization(TestCase):

    # 定义一个测试方法 test_nocedal_example，测试 Nocedal/Wright 书中的例子
    def test_nocedal_example(self):
        # 创建一个稀疏矩阵 H
        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        # 创建一个稀疏矩阵 A
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        # 创建一个 NumPy 数组 c
        c = np.array([-8, -3, -3])
        # 创建一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 调用 eqp_kktfact 函数，求解 KKT 系统
        x, lagrange_multipliers = eqp_kktfact(H, c, A, b)
        # 断言 x 数组与给定值接近
        assert_array_almost_equal(x, [2, -1, 1])
        # 断言 lagrange_multipliers 数组与给定值接近
        assert_array_almost_equal(lagrange_multipliers, [3, -2])


# 定义一个测试类 TestSphericalBoundariesIntersections，继承自 TestCase 类
class TestSphericalBoundariesIntersections(TestCase):

    # 定义一个测试方法 test_2d_sphere_constraints，测试二维球面约束条件
    def test_2d_sphere_constraints(self):
        # 测试内部初始点的球面交点计算
        ta, tb, intersect = sphere_intersections([0, 0],
                                                 [1, 0], 0.5)
        # 断言 ta 和 tb 的值接近给定值
        assert_array_almost_equal([ta, tb], [0, 0.5])
        # 断言 intersect 为 True
        assert_equal(intersect, True)

        # 测试直线与圆之间无交点的情况
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [0, 1], 1)
        # 断言 intersect 为 False
        assert_equal(intersect, False)

        # 测试初始点外部，指向圆外部的情况
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 1)
        # 断言 intersect 为 False
        assert_equal(intersect, False)

        # 测试初始点外部，指向圆内部的情况
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [-1, 0], 1.5)
        # 断言 ta 和 tb 的值接近给定值
        assert_array_almost_equal([ta, tb], [0.5, 1])
        # 断言 intersect 为 True
        assert_equal(intersect, True)

        # 测试初始点在边界上的情况
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 2)
        # 断言 ta 和 tb 的值接近给定值
        assert_array_almost_equal([ta, tb], [0, 0])
        # 断言 intersect 为 True
        assert_equal(intersect, True)
    # 定义测试函数，用于测试二维球体与直线的交点计算函数
    def test_2d_sphere_constraints_line_intersections(self):
        # 内部初始点测试
        ta, tb, intersect = sphere_intersections([0, 0],  # 球心坐标
                                                 [1, 0],  # 直线方向向量
                                                 0.5,     # 球半径
                                                 entire_line=True)  # 是否考虑整条直线
        assert_array_almost_equal([ta, tb], [-0.5, 0.5])  # 断言：期望的交点范围
        assert_equal(intersect, True)  # 断言：期望有交点

        # 直线与圆无交点测试
        ta, tb, intersect = sphere_intersections([2, 0],   # 球心坐标
                                                 [0, 1],   # 直线方向向量
                                                 1,        # 球半径
                                                 entire_line=True)  # 是否考虑整条直线
        assert_equal(intersect, False)  # 断言：期望无交点

        # 外部初始点朝向圆外测试
        ta, tb, intersect = sphere_intersections([2, 0],   # 球心坐标
                                                 [1, 0],   # 直线方向向量
                                                 1,        # 球半径
                                                 entire_line=True)  # 是否考虑整条直线
        assert_array_almost_equal([ta, tb], [-3, -1])  # 断言：期望的交点范围
        assert_equal(intersect, True)  # 断言：期望有交点

        # 外部初始点朝向圆内测试
        ta, tb, intersect = sphere_intersections([2, 0],   # 球心坐标
                                                 [-1, 0],  # 直线方向向量
                                                 1.5,      # 球半径
                                                 entire_line=True)  # 是否考虑整条直线
        assert_array_almost_equal([ta, tb], [0.5, 3.5])  # 断言：期望的交点范围
        assert_equal(intersect, True)  # 断言：期望有交点

        # 初始点在圆边界上测试
        ta, tb, intersect = sphere_intersections([2, 0],   # 球心坐标
                                                 [1, 0],   # 直线方向向量
                                                 2,        # 球半径
                                                 entire_line=True)  # 是否考虑整条直线
        assert_array_almost_equal([ta, tb], [-4, 0])  # 断言：期望的交点范围
        assert_equal(intersect, True)  # 断言：期望有交点
class
    # 定义一个测试方法，用于测试三维框约束函数的各种情况

    # 测试简单情况：两个框相交
    ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, 1],
                                          [1, 1, 1], [3, 3, 3])
    # 断言计算出的交点参数列表与预期的参数列表几乎相等
    assert_array_almost_equal([ta, tb], [1, 1])
    # 断言是否相交的布尔值为 True
    assert_equal(intersect, True)

    # 测试负方向的情况：两个框不相交
    ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, -1],
                                          [1, 1, 1], [3, 3, 3])
    # 断言是否相交的布尔值为 False
    assert_equal(intersect, False)

    # 测试内部点的情况：一个框包含在另一个框内部
    ta, tb, intersect = box_intersections([2, 2, 2], [0, -1, 1],
                                          [1, 1, 1], [3, 3, 3])
    # 断言计算出的交点参数列表与预期的参数列表几乎相等
    assert_array_almost_equal([ta, tb], [0, 1])
    # 断言是否相交的布尔值为 True

    # 定义另一个测试方法，用于测试三维框约束函数的整条线的情况

    # 测试简单情况：两个框相交
    ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, 1],
                                          [1, 1, 1], [3, 3, 3],
                                          entire_line=True)
    # 断言计算出的交点参数列表与预期的参数列表几乎相等
    assert_array_almost_equal([ta, tb], [1, 3])
    # 断言是否相交的布尔值为 True

    # 测试负方向的情况：两个框相交
    ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, -1],
                                          [1, 1, 1], [3, 3, 3],
                                          entire_line=True)
    # 断言计算出的交点参数列表与预期的参数列表几乎相等
    assert_array_almost_equal([ta, tb], [-3, -1])
    # 断言是否相交的布尔值为 True

    # 测试内部点的情况：一个框包含在另一个框内部
    ta, tb, intersect = box_intersections([2, 2, 2], [0, -1, 1],
                                          [1, 1, 1], [3, 3, 3],
                                          entire_line=True)
    # 断言计算出的交点参数列表与预期的参数列表几乎相等
    assert_array_almost_equal([ta, tb], [-1, 1])
    # 断言是否相交的布尔值为 True
class TestBoxSphereBoundariesIntersections(TestCase):

    def test_2d_box_constraints(self):
        # 测试两个约束都是活跃的情况
        ta, tb, intersect = box_sphere_intersections([1, 1], [-2, 2],
                                                     [-1, -2], [1, 2], 2,
                                                     entire_line=False)
        # 断言近似数组 [ta, tb] 应该为 [0, 0.5]
        assert_array_almost_equal([ta, tb], [0, 0.5])
        # 断言 intersect 应该为 True
        assert_equal(intersect, True)

        # 测试两个约束都不是活跃的情况
        ta, tb, intersect = box_sphere_intersections([1, 1], [-1, 1],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=False)
        # 断言近似数组 [ta, tb] 应该为 [0, 1]
        assert_array_almost_equal([ta, tb], [0, 1])
        # 断言 intersect 应该为 True
        assert_equal(intersect, True)

        # 测试盒子约束是活跃的情况
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=False)
        # 断言近似数组 [ta, tb] 应该为 [0, 0.5]
        assert_array_almost_equal([ta, tb], [0, 0.5])
        # 断言 intersect 应该为 True
        assert_equal(intersect, True)

        # 测试球形约束是活跃的情况
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=False)
        # 断言近似数组 [ta, tb] 应该为 [0, 0.25]
        assert_array_almost_equal([ta, tb], [0, 0.25])
        # 断言 intersect 应该为 True
        assert_equal(intersect, True)

        # 测试不可行的问题
        ta, tb, intersect = box_sphere_intersections([2, 2], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=False)
        # 断言 intersect 应该为 False
        assert_equal(intersect, False)
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [2, 4], [2, 4], 2,
                                                     entire_line=False)
        # 断言 intersect 应该为 False
        assert_equal(intersect, False)
    # 测试2D情况下盒子和球体约束的交点计算，考虑整条直线的情况

    # 情况1：两个约束都生效
    ta, tb, intersect = box_sphere_intersections([1, 1], [-2, 2],
                                                 [-1, -2], [1, 2], 2,
                                                 entire_line=True)
    # 断言计算出的交点比较接近期望值
    assert_array_almost_equal([ta, tb], [0, 0.5])
    # 断言两个对象是否相等
    assert_equal(intersect, True)

    # 情况2：两个约束都不生效
    ta, tb, intersect = box_sphere_intersections([1, 1], [-1, 1],
                                                 [-1, -3], [1, 3], 10,
                                                 entire_line=True)
    # 断言计算出的交点比较接近期望值
    assert_array_almost_equal([ta, tb], [0, 2])
    # 断言两个对象是否相等
    assert_equal(intersect, True)

    # 情况3：盒子约束生效
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                 [-1, -3], [1, 3], 10,
                                                 entire_line=True)
    # 断言计算出的交点比较接近期望值
    assert_array_almost_equal([ta, tb], [0, 0.5])
    # 断言两个对象是否相等
    assert_equal(intersect, True)

    # 情况4：球体约束生效
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                 [-1, -3], [1, 3], 2,
                                                 entire_line=True)
    # 断言计算出的交点比较接近期望值
    assert_array_almost_equal([ta, tb], [0, 0.25])
    # 断言两个对象是否相等
    assert_equal(intersect, True)

    # 情况5：问题不可行
    ta, tb, intersect = box_sphere_intersections([2, 2], [-4, 4],
                                                 [-1, -3], [1, 3], 2,
                                                 entire_line=True)
    # 断言对象是否相等，即交点是否为False
    assert_equal(intersect, False)

    # 情况6：问题不可行
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                 [2, 4], [2, 4], 2,
                                                 entire_line=True)
    # 断言对象是否相等，即交点是否为False
    assert_equal(intersect, False)
class TestModifiedDogleg(TestCase):

    def test_cauchypoint_equalsto_newtonpoint(self):
        A = np.array([[1, 8]])  # 定义一个2维数组A
        b = np.array([-16])  # 定义一个包含单个元素-16的数组b
        _, _, Y = projections(A)  # 调用projections函数，获取Y值

        newton_point = np.array([0.24615385, 1.96923077])  # 定义一个新顿点坐标数组

        # Newton point inside boundaries
        # 使用modified_dogleg函数计算新顿点在指定边界内的值x
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [np.inf, np.inf])
        assert_array_almost_equal(x, newton_point)  # 断言x与预期的新顿点坐标相近

        # Spherical constraint active
        # 使用modified_dogleg函数计算在球形约束条件下的值x
        x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf], [np.inf, np.inf])
        assert_array_almost_equal(x, newton_point/np.linalg.norm(newton_point))  # 断言x与新顿点的归一化值相近

        # Box constraints active
        # 使用modified_dogleg函数计算在盒形约束条件下的值x
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [0.1, np.inf])
        assert_array_almost_equal(x, (newton_point/newton_point[0]) * 0.1)  # 断言x与按比例缩放的新顿点坐标相近
    # 定义一个测试方法，用于测试三维例子
    def test_3d_example(self):
        # 创建一个2x3的NumPy数组A
        A = np.array([[1, 8, 1],
                      [4, 2, 2]])
        # 创建一个包含两个元素的NumPy数组b
        b = np.array([-16, 2])
        # 调用 projections 函数，并返回三个变量 Z, LS, Y
        Z, LS, Y = projections(A)

        # 设置一个包含三个元素的新ton_point变量
        newton_point = np.array([-1.37090909, 2.23272727, -0.49090909])
        # 设置一个包含三个元素的cauchy_point变量
        cauchy_point = np.array([0.11165723, 1.73068711, 0.16748585])
        # 创建一个与newton_point相同形状的全零数组origin
        origin = np.zeros_like(newton_point)

        # 调用 modified_dogleg 函数，计算满足条件的最佳点x，并断言其与newton_point几乎相等
        x = modified_dogleg(A, Y, b, 3, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        assert_array_almost_equal(x, newton_point)

        # 调用 modified_dogleg 函数，计算满足条件的最佳点x，并断言其与newton_point的相对距离与期望值几乎相等
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = cauchy_point
        d = newton_point - cauchy_point
        t = ((x - z) / (d))
        assert_array_almost_equal(t, np.full(3, 0.40807330))
        assert_array_almost_equal(np.linalg.norm(x), 2)

        # 调用 modified_dogleg 函数，计算满足条件的最佳点x，并断言其与cauchy_point的相对距离与期望值几乎相等
        x = modified_dogleg(A, Y, b, 5, [-1, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = cauchy_point
        d = newton_point - cauchy_point
        t = ((x - z) / (d))
        assert_array_almost_equal(t, np.full(3, 0.7498195))
        assert_array_almost_equal(x[0], -1)

        # 调用 modified_dogleg 函数，计算满足条件的最佳点x，并断言其与origin的相对距离与期望值几乎相等
        x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = origin
        d = cauchy_point
        t = ((x - z) / (d))
        assert_array_almost_equal(t, np.full(3, 0.573936265))
        assert_array_almost_equal(np.linalg.norm(x), 1)

        # 调用 modified_dogleg 函数，计算满足条件的最佳点x，并断言其与origin的相对距离与期望值几乎相等
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf],
                            [np.inf, 1, np.inf])
        z = origin
        d = newton_point
        t = ((x - z) / (d))
        assert_array_almost_equal(t, np.full(3, 0.4478827364))
        assert_array_almost_equal(x[1], 1)
# 定义一个测试类 TestProjectCG，用于测试项目CG的功能
class TestProjectCG(TestCase):

    # 示例来源于《Numerical Optimization》第452页，Nocedal/Wright的例子
    def test_nocedal_example(self):
        # 创建一个稀疏压缩列（CSC）格式的矩阵 H
        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        # 创建一个稀疏压缩列（CSC）格式的矩阵 A
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        # 创建一个 NumPy 数组 c
        c = np.array([-8, -3, -3])
        # 创建一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 调用 projections 函数，获取返回值 Z, _, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数，进行共轭梯度优化，获取优化结果 x 和信息字典 info
        x, info = projected_cg(H, c, Z, Y, b)
        # 断言优化结束条件为 4
        assert_equal(info["stop_cond"], 4)
        # 断言未触及边界
        assert_equal(info["hits_boundary"], False)
        # 断言优化结果 x 与预期结果接近
        assert_array_almost_equal(x, [2, -1, 1])

    # 测试与直接因子化方法比较
    def test_compare_with_direct_fact(self):
        # 创建一个稀疏压缩列（CSC）格式的矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 创建一个稀疏压缩列（CSC）格式的矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 创建一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 调用 projections 函数，获取返回值 Z, _, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数，进行共轭梯度优化，获取优化结果 x 和信息字典 info，设置公差 tol=0
        x, info = projected_cg(H, c, Z, Y, b, tol=0)
        # 调用 eqp_kktfact 函数，获取直接因子化方法的结果 x_kkt
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        # 断言优化结束条件为 1
        assert_equal(info["stop_cond"], 1)
        # 断言未触及边界
        assert_equal(info["hits_boundary"], False)
        # 断言优化结果 x 与直接因子化结果 x_kkt 接近
        assert_array_almost_equal(x, x_kkt)

    # 测试信任域算法不可行情况
    def test_trust_region_infeasible(self):
        # 创建一个稀疏压缩列（CSC）格式的矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 创建一个稀疏压缩列（CSC）格式的矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 创建一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 设定信任域半径为 1
        trust_radius = 1
        # 调用 projections 函数，获取返回值 Z, _, Y
        Z, _, Y = projections(A)
        # 使用 pytest 的异常断言，测试 projected_cg 函数在指定的信任域半径下是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, trust_radius=trust_radius)

    # 测试信任域算法边缘可行情况
    def test_trust_region_barely_feasible(self):
        # 创建一个稀疏压缩列（CSC）格式的矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 创建一个稀疏压缩列（CSC）格式的矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 创建一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 设定信任域半径为 2.32379000772445021283
        trust_radius = 2.32379000772445021283
        # 调用 projections 函数，获取返回值 Z, _, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数，进行共轭梯度优化，获取优化结果 x 和信息字典 info，设置公差 tol=0，信任域半径 trust_radius
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        # 断言优化结束条件为 2
        assert_equal(info["stop_cond"], 2)
        # 断言触及边界
        assert_equal(info["hits_boundary"], True)
        # 断言优化结果 x 的二范数与设定的信任域半径 trust_radius 接近
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)
        # 断言优化结果 x 与期望结果 -Y.dot(b) 接近
        assert_array_almost_equal(x, -Y.dot(b))
    # 定义一个测试函数，用于测试在存在边界条件下的情况
    def test_hits_boundary(self):
        # 创建一个稀疏压缩列矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 创建一个稀疏压缩列矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 创建一个 numpy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 numpy 数组 b
        b = -np.array([3, 0])
        # 设置信任半径
        trust_radius = 3
        # 对矩阵 A 进行投影操作，获取投影后的结果 Z, _, Y
        Z, _, Y = projections(A)
        # 使用投影共轭梯度法求解优化问题，返回优化结果 x 和信息 info
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        # 断言优化停止条件为 2
        assert_equal(info["stop_cond"], 2)
        # 断言 hits_boundary 字段为 True
        assert_equal(info["hits_boundary"], True)
        # 断言计算得到的 x 的范数近似等于设定的信任半径
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    # 测试存在负曲率情况下的无约束优化问题
    def test_negative_curvature_unconstrained(self):
        # 创建一个稀疏压缩列矩阵 H
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        # 创建一个稀疏压缩列矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        # 创建一个 numpy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 numpy 数组 b
        b = -np.array([3, 0])
        # 对矩阵 A 进行投影操作，获取投影后的结果 Z, _, Y
        Z, _, Y = projections(A)
        # 使用投影共轭梯度法求解优化问题，预期会抛出 ValueError
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, tol=0)

    # 测试存在负曲率情况下的优化问题
    def test_negative_curvature(self):
        # 创建一个稀疏压缩列矩阵 H
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        # 创建一个稀疏压缩列矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        # 创建一个 numpy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 numpy 数组 b
        b = -np.array([3, 0])
        # 对矩阵 A 进行投影操作，获取投影后的结果 Z, _, Y
        Z, _, Y = projections(A)
        # 设置较大的信任半径
        trust_radius = 1000
        # 使用投影共轭梯度法求解优化问题，返回优化结果 x 和信息 info
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        # 断言优化停止条件为 3
        assert_equal(info["stop_cond"], 3)
        # 断言 hits_boundary 字段为 True
        assert_equal(info["hits_boundary"], True)
        # 断言计算得到的 x 的范数近似等于设定的信任半径
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    # 在解处，箱约束是不活跃的，但在迭代过程中是活跃的
    def test_inactive_box_constraints(self):
        # 创建一个稀疏压缩列矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 创建一个稀疏压缩列矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 创建一个 numpy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个 numpy 数组 b
        b = -np.array([3, 0])
        # 对矩阵 A 进行投影操作，获取投影后的结果 Z, _, Y
        Z, _, Y = projections(A)
        # 使用投影共轭梯度法求解优化问题，返回优化结果 x 和信息 info
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               lb=[0.5, -np.inf,
                                   -np.inf, -np.inf],
                               return_all=True)
        # 使用等式对问题进行 KKT 因子化，得到解 x_kkt
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        # 断言优化停止条件为 1
        assert_equal(info["stop_cond"], 1)
        # 断言 hits_boundary 字段为 False
        assert_equal(info["hits_boundary"], False)
        # 断言计算得到的 x 与 KKT 等式求解得到的 x_kkt 近似相等
        assert_array_almost_equal(x, x_kkt)

    # 箱约束处于活跃状态，终止条件是最大迭代次数（非可行交互）
    # 定义一个测试函数，测试当活动框约束条件下达到最大迭代次数的情况
    def test_active_box_constraints_maximum_iterations_reached(self):
        # 定义一个稀疏矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 定义一个稀疏矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 定义一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 定义一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 调用 projections 函数获取 Z, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数进行投影共轭梯度计算
        # 返回 x 和 info，其中设置了最大迭代次数、下界以及返回所有迭代过程
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               lb=[0.8, -np.inf,
                                   -np.inf, -np.inf],
                               return_all=True)
        # 断言检查停止条件为 1
        assert_equal(info["stop_cond"], 1)
        # 断言检查是否命中边界
        assert_equal(info["hits_boundary"], True)
        # 断言检查计算得到的 Ax 是否接近于 -b
        assert_array_almost_equal(A.dot(x), -b)
        # 断言检查 x[0] 是否接近于 0.8
        assert_array_almost_equal(x[0], 0.8)

    # 活动框约束条件下，终止因为命中边界（无非可行交互）的测试函数
    def test_active_box_constraints_hits_boundaries(self):
        # 定义一个稀疏矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 定义一个稀疏矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 定义一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 定义一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 定义信任半径为 3
        trust_radius = 3
        # 调用 projections 函数获取 Z, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数进行投影共轭梯度计算
        # 返回 x 和 info，其中设置了最大迭代次数、上界、信任半径以及返回所有迭代过程
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, np.inf, 1.6, np.inf],
                               trust_radius=trust_radius,
                               return_all=True)
        # 断言检查停止条件为 2
        assert_equal(info["stop_cond"], 2)
        # 断言检查是否命中边界
        assert_equal(info["hits_boundary"], True)
        # 断言检查 x[2] 是否接近于 1.6
        assert_array_almost_equal(x[2], 1.6)

    # 活动框约束条件下，终止因为命中边界（有非可行交互）的测试函数
    def test_active_box_constraints_hits_boundaries_infeasible_iter(self):
        # 定义一个稀疏矩阵 H
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        # 定义一个稀疏矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        # 定义一个 NumPy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 定义一个 NumPy 数组 b
        b = -np.array([3, 0])
        # 定义信任半径为 4
        trust_radius = 4
        # 调用 projections 函数获取 Z, Y
        Z, _, Y = projections(A)
        # 调用 projected_cg 函数进行投影共轭梯度计算
        # 返回 x 和 info，其中设置了最大迭代次数、上界、信任半径以及返回所有迭代过程
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, 0.1, np.inf, np.inf],
                               trust_radius=trust_radius,
                               return_all=True)
        # 断言检查停止条件为 2
        assert_equal(info["stop_cond"], 2)
        # 断言检查是否命中边界
        assert_equal(info["hits_boundary"], True)
        # 断言检查 x[1] 是否接近于 0.1
        assert_array_almost_equal(x[1], 0.1)

    # 活动框约束条件下，终止因为命中边界（无非可行交互）的测试函数
    def test_active_box_constraints_hits_boundaries_no_infeasible(self):
        # 此函数暂时没有代码，只是注释。
    # 定义一个测试函数，用于测试在负曲率情况下的活动边界约束
    def test_active_box_constraints_negative_curvature(self):
        # 创建一个稀疏矩阵 H
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        # 创建一个稀疏矩阵 A
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        # 创建一个包含四个元素的 numpy 数组 c
        c = np.array([-2, -3, -3, 1])
        # 创建一个包含两个元素的 numpy 数组 b，其值为负数
        b = -np.array([3, 0])
        # 使用 projections 函数计算投影 Z 和 Y
        Z, _, Y = projections(A)
        # 设置信任半径为 1000
        trust_radius = 1000
        # 调用 projected_cg 函数，使用共轭梯度法求解优化问题
        # x 是优化结果，info 是优化过程中的信息
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, np.inf, 100, np.inf],
                               trust_radius=trust_radius)
        # 断言优化过程停止条件为 3
        assert_equal(info["stop_cond"], 3)
        # 断言优化过程中是否触及边界
        assert_equal(info["hits_boundary"], True)
        # 断言第三个元素 x[2] 的值接近于 100
        assert_array_almost_equal(x[2], 100)
```