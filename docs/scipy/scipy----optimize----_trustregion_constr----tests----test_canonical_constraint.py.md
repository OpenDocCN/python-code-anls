# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tests\test_canonical_constraint.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_array_equal, assert_equal  # 导入 NumPy 测试模块中的数组比较函数
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,  # 导入 SciPy 优化模块中的约束相关类
                                         PreparedConstraint)
from scipy.optimize._trustregion_constr.canonical_constraint \
    import CanonicalConstraint, initial_constraints_as_canonical  # 导入 SciPy 优化模块中的 CanonicalConstraint 类和相关函数


def create_quadratic_function(n, m, rng):
    a = rng.rand(m)  # 随机生成 m 个元素的数组 a
    A = rng.rand(m, n)  # 随机生成大小为 m x n 的数组 A
    H = rng.rand(m, n, n)  # 随机生成大小为 m x n x n 的数组 H
    HT = np.transpose(H, (1, 2, 0))  # 将 H 按指定轴重新排列成 HT

    def fun(x):
        return a + A.dot(x) + 0.5 * H.dot(x).dot(x)  # 定义二次函数 fun(x)

    def jac(x):
        return A + H.dot(x)  # 定义 fun(x) 的雅可比矩阵 jac(x)

    def hess(x, v):
        return HT.dot(v)  # 定义 fun(x) 的黑塞矩阵 hess(x, v)

    return fun, jac, hess  # 返回函数 fun, jac 和 hess


def test_bounds_cases():
    # Test 1: no constraints.
    user_constraint = Bounds(-np.inf, np.inf)  # 创建无约束的边界对象
    x0 = np.array([-1, 2])  # 初始点 x0
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)  # 使用边界对象创建准备约束对象
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)  # 转换为 CanonicalConstraint 对象

    assert_equal(c.n_eq, 0)  # 断言等式约束数为0
    assert_equal(c.n_ineq, 0)  # 断言不等式约束数为0

    c_eq, c_ineq = c.fun(x0)  # 计算在 x0 处的等式约束和不等式约束
    assert_array_equal(c_eq, [])  # 断言等式约束结果为空数组
    assert_array_equal(c_ineq, [])  # 断言不等式约束结果为空数组

    J_eq, J_ineq = c.jac(x0)  # 计算在 x0 处的等式约束和不等式约束的雅可比矩阵
    assert_array_equal(J_eq, np.empty((0, 2)))  # 断言等式约束的雅可比矩阵为空矩阵
    assert_array_equal(J_ineq, np.empty((0, 2)))  # 断言不等式约束的雅可比矩阵为空矩阵

    assert_array_equal(c.keep_feasible, [])  # 断言保持可行性的数组为空

    # Test 2: infinite lower bound.
    user_constraint = Bounds(-np.inf, [0, np.inf, 1], [False, True, True])  # 创建具有无限下界的边界对象
    x0 = np.array([-1, -2, -3], dtype=float)  # 初始点 x0
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)  # 使用边界对象创建准备约束对象
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)  # 转换为 CanonicalConstraint 对象

    assert_equal(c.n_eq, 0)  # 断言等式约束数为0
    assert_equal(c.n_ineq, 2)  # 断言不等式约束数为2

    c_eq, c_ineq = c.fun(x0)  # 计算在 x0 处的等式约束和不等式约束
    assert_array_equal(c_eq, [])  # 断言等式约束结果为空数组
    assert_array_equal(c_ineq, [-1, -4])  # 断言不等式约束结果为 [-1, -4]

    J_eq, J_ineq = c.jac(x0)  # 计算在 x0 处的等式约束和不等式约束的雅可比矩阵
    assert_array_equal(J_eq, np.empty((0, 3)))  # 断言等式约束的雅可比矩阵为空矩阵
    assert_array_equal(J_ineq, np.array([[1, 0, 0], [0, 0, 1]]))  # 断言不等式约束的雅可比矩阵为指定数组

    assert_array_equal(c.keep_feasible, [False, True])  # 断言保持可行性的数组为指定值

    # Test 3: infinite upper bound.
    user_constraint = Bounds([0, 1, -np.inf], np.inf, [True, False, True])  # 创建具有无限上界的边界对象
    x0 = np.array([1, 2, 3], dtype=float)  # 初始点 x0
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)  # 使用边界对象创建准备约束对象
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)  # 转换为 CanonicalConstraint 对象

    assert_equal(c.n_eq, 0)  # 断言等式约束数为0
    assert_equal(c.n_ineq, 2)  # 断言不等式约束数为2

    c_eq, c_ineq = c.fun(x0)  # 计算在 x0 处的等式约束和不等式约束
    assert_array_equal(c_eq, [])  # 断言等式约束结果为空数组
    assert_array_equal(c_ineq, [-1, -1])  # 断言不等式约束结果为 [-1, -1]

    J_eq, J_ineq = c.jac(x0)  # 计算在 x0 处的等式约束和不等式约束的雅可比矩阵
    assert_array_equal(J_eq, np.empty((0, 3)))  # 断言等式约束的雅可比矩阵为空矩阵
    assert_array_equal(J_ineq, np.array([[-1, 0, 0], [0, -1, 0]]))  # 断言不等式约束的雅可比矩阵为指定数组

    assert_array_equal(c.keep_feasible, [True, False])  # 断言保持可行性的数组为指定值

    # Test 4: interval constraint.
    user_constraint = Bounds([-1, -np.inf, 2, 3], [1, np.inf, 10, 3],
                             [False, True, True, True])  # 创建具有区间约束的边界对象
    x0 = np.array([0, 10, 8, 5])  # 初始点 x0
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)  # 使用边界对象创建准备约束对象
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)  # 转换为 CanonicalConstraint 对象

    assert_equal(c.n_eq, 1)  # 断言等式约束数为1
    # 断言 c 对象的 n_ineq 属性是否等于 4
    assert_equal(c.n_ineq, 4)
    
    # 调用 c 对象的 fun 方法，传入参数 x0，返回 c_eq 和 c_ineq 两个数组
    c_eq, c_ineq = c.fun(x0)
    # 断言 c_eq 数组是否等于 [2]
    assert_array_equal(c_eq, [2])
    # 断言 c_ineq 数组是否等于 [-1, -2, -1, -6]
    assert_array_equal(c_ineq, [-1, -2, -1, -6])
    
    # 调用 c 对象的 jac 方法，传入参数 x0，返回 J_eq 和 J_ineq 两个二维数组
    J_eq, J_ineq = c.jac(x0)
    # 断言 J_eq 二维数组是否等于 [[0, 0, 0, 1]]
    assert_array_equal(J_eq, [[0, 0, 0, 1]])
    # 断言 J_ineq 二维数组是否等于 [[1, 0, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, -1, 0]]
    assert_array_equal(J_ineq, [[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [-1, 0, 0, 0],
                                [0, 0, -1, 0]])
    
    # 断言 c 对象的 keep_feasible 属性是否等于 [False, True, False, True]
    assert_array_equal(c.keep_feasible, [False, True, False, True])
def test_nonlinear_constraint():
    n = 3  # 定义变量维度
    m = 5  # 定义约束条件数量
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    x0 = rng.rand(n)  # 生成随机初始向量

    fun, jac, hess = create_quadratic_function(n, m, rng)  # 调用函数创建二次函数及其导数
    f = fun(x0)  # 计算函数值
    J = jac(x0)  # 计算雅可比矩阵

    lb = [-10, 3, -np.inf, -np.inf, -5]  # 设置下界
    ub = [10, 3, np.inf, 3, np.inf]  # 设置上界
    user_constraint = NonlinearConstraint(
        fun, lb, ub, jac, hess, [True, False, False, True, False])  # 创建非线性约束对象

    for sparse_jacobian in [False, True]:  # 循环处理稀疏雅可比矩阵的情况
        prepared_constraint = PreparedConstraint(user_constraint, x0,
                                                 sparse_jacobian)  # 准备约束条件对象
        c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)  # 从准备的约束条件创建规范化约束对象

        assert_array_equal(c.n_eq, 1)  # 断言相等：等式约束数目应为1
        assert_array_equal(c.n_ineq, 4)  # 断言相等：不等式约束数目应为4

        c_eq, c_ineq = c.fun(x0)  # 计算等式约束和不等式约束的值
        assert_array_equal(c_eq, [f[1] - lb[1]])  # 断言相等：等式约束的值应为 f[1] - lb[1]
        assert_array_equal(c_ineq, [f[3] - ub[3], lb[4] - f[4],
                                    f[0] - ub[0], lb[0] - f[0]])  # 断言相等：各不等式约束的值应为对应的数值

        J_eq, J_ineq = c.jac(x0)  # 计算等式约束和不等式约束的雅可比矩阵
        if sparse_jacobian:
            J_eq = J_eq.toarray()  # 如果稀疏雅可比矩阵，则转换为密集型
            J_ineq = J_ineq.toarray()  # 如果稀疏雅可比矩阵，则转换为密集型

        assert_array_equal(J_eq, J[1, None])  # 断言相等：等式约束的雅可比矩阵应与预期值相符
        assert_array_equal(J_ineq, np.vstack((J[3], -J[4], J[0], -J[0])))  # 断言相等：不等式约束的雅可比矩阵应与预期值相符

        v_eq = rng.rand(c.n_eq)  # 生成随机等式约束向量
        v_ineq = rng.rand(c.n_ineq)  # 生成随机不等式约束向量
        v = np.zeros(m)  # 初始化向量
        v[1] = v_eq[0]  # 设置等式约束向量的值
        v[3] = v_ineq[0]  # 设置不等式约束向量的值
        v[4] = -v_ineq[1]  # 设置不等式约束向量的值
        v[0] = v_ineq[2] - v_ineq[3]  # 设置不等式约束向量的值
        assert_array_equal(c.hess(x0, v_eq, v_ineq), hess(x0, v))  # 断言相等：Hessian 矩阵应与预期值相符

        assert_array_equal(c.keep_feasible, [True, False, True, True])  # 断言相等：保持可行性的设置应与预期相符


def test_concatenation():
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    n = 4  # 定义变量维度
    x0 = rng.rand(n)  # 生成随机初始向量

    f1 = x0  # 定义函数值
    J1 = np.eye(n)  # 定义单位矩阵的雅可比矩阵
    lb1 = [-1, -np.inf, -2, 3]  # 设置第一组约束的下界
    ub1 = [1, np.inf, np.inf, 3]  # 设置第一组约束的上界
    bounds = Bounds(lb1, ub1, [False, False, True, False])  # 创建约束对象

    fun, jac, hess = create_quadratic_function(n, 5, rng)  # 调用函数创建二次函数及其导数
    f2 = fun(x0)  # 计算函数值
    J2 = jac(x0)  # 计算雅可比矩阵
    lb2 = [-10, 3, -np.inf, -np.inf, -5]  # 设置第二组约束的下界
    ub2 = [10, 3, np.inf, 5, np.inf]  # 设置第二组约束的上界
    nonlinear = NonlinearConstraint(
        fun, lb2, ub2, jac, hess, [True, False, False, True, False])  # 创建非线性约束对象
    # 对于每个稀疏雅可比矩阵的情况进行循环，分别为 False 和 True
    for sparse_jacobian in [False, True]:
        # 准备边界约束对象
        bounds_prepared = PreparedConstraint(bounds, x0, sparse_jacobian)
        # 准备非线性约束对象
        nonlinear_prepared = PreparedConstraint(nonlinear, x0, sparse_jacobian)

        # 根据准备好的边界约束对象创建规范约束对象 c1
        c1 = CanonicalConstraint.from_PreparedConstraint(bounds_prepared)
        # 根据准备好的非线性约束对象创建规范约束对象 c2
        c2 = CanonicalConstraint.from_PreparedConstraint(nonlinear_prepared)
        # 将 c1 和 c2 连接成一个新的规范约束对象 c
        c = CanonicalConstraint.concatenate([c1, c2], sparse_jacobian)

        # 断言规范约束对象 c 的等式约束数目为 2
        assert_equal(c.n_eq, 2)
        # 断言规范约束对象 c 的不等式约束数目为 7
        assert_equal(c.n_ineq, 7)

        # 计算规范约束对象 c 在初始点 x0 处的等式约束和不等式约束值
        c_eq, c_ineq = c.fun(x0)
        # 断言计算得到的等式约束值 c_eq 符合预期值 [f1[3] - lb1[3], f2[1] - lb2[1]]
        assert_array_equal(c_eq, [f1[3] - lb1[3], f2[1] - lb2[1]])
        # 断言计算得到的不等式约束值 c_ineq 符合预期值
        assert_array_equal(c_ineq, [lb1[2] - f1[2], f1[0] - ub1[0],
                                    lb1[0] - f1[0], f2[3] - ub2[3],
                                    lb2[4] - f2[4], f2[0] - ub2[0],
                                    lb2[0] - f2[0]])

        # 计算规范约束对象 c 在初始点 x0 处的雅可比矩阵 J_eq 和 J_ineq
        J_eq, J_ineq = c.jac(x0)
        # 如果使用稀疏雅可比矩阵，则将 J_eq 和 J_ineq 转换为稠密数组
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        # 断言计算得到的等式约束的雅可比矩阵 J_eq 符合预期值 np.vstack((J1[3], J2[1]))
        assert_array_equal(J_eq, np.vstack((J1[3], J2[1])))
        # 断言计算得到的不等式约束的雅可比矩阵 J_ineq 符合预期值
        assert_array_equal(J_ineq, np.vstack((-J1[2], J1[0], -J1[0], J2[3],
                                              -J2[4], J2[0], -J2[0])))

        # 生成与规范约束对象 c 相关的等式约束和不等式约束的随机向量 v_eq 和 v_ineq
        v_eq = rng.rand(c.n_eq)
        v_ineq = rng.rand(c.n_ineq)
        v = np.zeros(5)
        # 设置 v 的第1个、第3个和第4个元素分别为 v_eq 和 v_ineq 的对应元素
        v[1] = v_eq[1]
        v[3] = v_ineq[3]
        v[4] = -v_ineq[4]
        # 计算用于 Hessian 矩阵计算的向量 v 的第0个元素
        v[0] = v_ineq[5] - v_ineq[6]
        # 计算规范约束对象 c 在初始点 x0 处及向量 v_eq、v_ineq 下的 Hessian 矩阵
        H = c.hess(x0, v_eq, v_ineq).dot(np.eye(n))
        # 断言计算得到的 Hessian 矩阵 H 符合预期的 hess(x0, v) 的值
        assert_array_equal(H, hess(x0, v))

        # 断言规范约束对象 c 的 keep_feasible 属性符合预期值
        assert_array_equal(c.keep_feasible,
                           [True, False, False, True, False, True, True])
def test_initial_constraints_as_canonical_empty():
    # 设置问题维度为3
    n = 3
    # 遍历是否使用稀疏雅可比矩阵的选项
    for sparse_jacobian in [False, True]:
        # 调用 initial_constraints_as_canonical 函数，传入空的约束列表和其他参数
        c_eq, c_ineq, J_eq, J_ineq = initial_constraints_as_canonical(
            n, [], sparse_jacobian)

        # 断言约束条件为零
        assert_array_equal(c_eq, [])
        assert_array_equal(c_ineq, [])

        # 如果使用稀疏雅可比矩阵，将稀疏雅可比矩阵转换为密集数组
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        # 断言雅可比矩阵为空矩阵
        assert_array_equal(J_eq, np.empty((0, n)))
        assert_array_equal(J_ineq, np.empty((0, n)))
```