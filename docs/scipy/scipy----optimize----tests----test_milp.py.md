# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_milp.py`

```
"""
Unit test for Mixed Integer Linear Programming
"""
# 导入正则表达式模块
import re

# 导入NumPy库及其测试模块
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
# 导入pytest测试框架
import pytest

# 导入线性规划测试模块
from .test_linprog import magic_square
# 导入混合整数线性规划函数及相关类
from scipy.optimize import milp, Bounds, LinearConstraint
# 导入稀疏矩阵处理库
from scipy import sparse


# 定义测试函数test_milp_iv
def test_milp_iv():

    # 定义错误信息字符串
    message = "`c` must be a dense array"
    # 使用pytest的raises断言，验证是否会引发特定错误
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入稀疏的COO格式数组，期望引发错误
        milp(sparse.coo_array([0, 0]))

    message = "`c` must be a one-dimensional array of finite numbers with"
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入非一维的零数组，期望引发错误
        milp(np.zeros((3, 4)))
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入空列表，期望引发错误
        milp([])
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入None，期望引发错误
        milp(None)

    message = "`bounds` must be convertible into an instance of..."
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入一个整数，期望引发错误
        milp(1, bounds=10)

    message = "`constraints` (or each element within `constraints`) must be"
    with pytest.raises(ValueError, match=re.escape(message)):
        # 调用milp函数并传入一个整数作为约束条件，期望引发错误
        milp(1, constraints=10)
    with pytest.raises(ValueError, match=re.escape(message)):
        # 调用milp函数并传入不符合要求的约束条件数组，期望引发错误
        milp(np.zeros(3), constraints=([[1, 2, 3]], [2, 3], [2, 3]))
    with pytest.raises(ValueError, match=re.escape(message)):
        # 调用milp函数并传入不符合要求的约束条件数组，期望引发错误
        milp(np.zeros(2), constraints=([[1, 2]], [2], sparse.coo_array([2])))

    message = "The shape of `A` must be (len(b_l), len(c))."
    with pytest.raises(ValueError, match=re.escape(message)):
        # 调用milp函数并传入不符合要求的约束条件数组，期望引发错误
        milp(np.zeros(3), constraints=([[1, 2]], [2], [2]))

    message = "`integrality` must be a dense array"
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入稀疏的COO格式数组，期望引发错误
        milp([1, 2], integrality=sparse.coo_array([1, 2]))

    message = ("`integrality` must contain integers 0-3 and be broadcastable "
               "to `c.shape`.")
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的整数性变量数组，期望引发错误
        milp([1, 2, 3], integrality=[1, 2])
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的整数性变量数组，期望引发错误
        milp([1, 2, 3], integrality=[1, 5, 3])

    message = "Lower and upper bounds must be dense arrays."
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入稀疏的COO格式数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2], sparse.coo_array([3, 4])))

    message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的边界数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2], [3, 4, 5]))
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的边界数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2, 3], [4, 5]))

    message = "`bounds.lb` and `bounds.ub` must contain reals and..."
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的边界数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2], [3, 4]))
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的边界数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2, 3], ["3+4", 4, 5]))
    with pytest.raises(ValueError, match=message):
        # 调用milp函数并传入不符合要求的边界数组，期望引发错误
        milp([1, 2, 3], bounds=([1, 2, 3], [set(), 4, 5]))


@pytest.mark.xfail(run=False,
                   reason="Needs to be fixed in `_highs_wrapper`")
# 定义测试函数test_milp_options，并标记为预期失败
def test_milp_options(capsys):
    # run=False now because of gh-16347
    # 定义警告消息，表示检测到未识别的选项
    message = "Unrecognized options detected: {'ekki'}..."
    # 设置选项字典，包含未识别的选项 'ekki'
    options = {'ekki': True}
    # 使用 pytest 的 warn 断言检测运行时警告，并匹配预期的警告消息
    with pytest.warns(RuntimeWarning, match=message):
        # 调用 milp 函数，传入参数 1 和设定的选项字典
        milp(1, options=options)

    # 调用 magic_square 函数，生成解析后的返回值
    A, b, c, numbers, M = magic_square(3)
    # 设置 MILP 函数的选项字典，包含显示、预处理关闭和时间限制为 0.05 秒
    options = {"disp": True, "presolve": False, "time_limit": 0.05}
    # 调用 milp 函数，传入约束矩阵 A, 不等式向量 b, 目标函数系数向量 c,
    # 变量上下界为 (0, 1), 整数变量标记 integrality=1, 和设定的选项字典
    res = milp(c=c, constraints=(A, b, b), bounds=(0, 1), integrality=1,
               options=options)

    # 读取并捕获 capsys 输出的内容
    captured = capsys.readouterr()
    # 断言输出中包含 "Presolve is switched off"
    assert "Presolve is switched off" in captured.out
    # 断言输出中包含 "Time Limit Reached"
    assert "Time Limit Reached" in captured.out
    # 断言 MILP 求解结果的成功标记为假
    assert not res.success
def test_result():
    # 调用 magic_square 函数生成问题的系数矩阵和约束条件
    A, b, c, numbers, M = magic_square(3)
    # 调用 milp 函数求解整数线性规划问题
    res = milp(c=c, constraints=(A, b, b), bounds=(0, 1), integrality=1)
    # 断言求解结果状态为成功
    assert res.status == 0
    assert res.success
    # 断言返回消息以指定字符串开头
    msg = "Optimization terminated successfully. (HiGHS Status 7:"
    assert res.message.startswith(msg)
    # 断言返回值类型为 numpy 数组和浮点数
    assert isinstance(res.x, np.ndarray)
    assert isinstance(res.fun, float)
    assert isinstance(res.mip_node_count, int)
    assert isinstance(res.mip_dual_bound, float)
    assert isinstance(res.mip_gap, float)

    # 重新调用 magic_square 函数生成新问题的系数矩阵和约束条件
    A, b, c, numbers, M = magic_square(6)
    # 调用 milp 函数，设置时间限制为 0.05 秒
    res = milp(c=c*0, constraints=(A, b, b), bounds=(0, 1), integrality=1,
               options={'time_limit': 0.05})
    # 断言求解结果状态为时间限制到达
    assert res.status == 1
    assert not res.success
    # 断言返回消息以指定字符串开头
    msg = "Time limit reached. (HiGHS Status 13:"
    assert res.message.startswith(msg)
    # 断言返回值为 None 类型
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)

    # 调用 milp 函数，提供不完整的参数
    res = milp(1, bounds=(1, -1))
    # 断言求解结果状态为问题不可行
    assert res.status == 2
    assert not res.success
    # 断言返回消息以指定字符串开头
    msg = "The problem is infeasible. (HiGHS Status 8:"
    assert res.message.startswith(msg)
    # 断言返回值为 None 类型
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)

    # 调用 milp 函数，提供不完整的参数
    res = milp(-1)
    # 断言求解结果状态为问题无界
    assert res.status == 3
    assert not res.success
    # 断言返回消息以指定字符串开头
    msg = "The problem is unbounded. (HiGHS Status 10:"
    assert res.message.startswith(msg)
    # 断言返回值为 None 类型
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)


def test_milp_optional_args():
    # 检查 milp 函数的非 c 参数确实是可选的
    res = milp(1)
    # 断言返回最优值为 0
    assert res.fun == 0
    # 断言返回的解向量为 [0]
    assert_array_equal(res.x, [0])


def test_milp_1():
    # 解决魔方问题
    n = 3
    A, b, c, numbers, M = magic_square(n)
    A = sparse.csc_array(A)  # 确认稀疏数组被接受
    # 调用 milp 函数解决整数线性规划问题
    res = milp(c=c*0, constraints=(A, b, b), bounds=(0, 1), integrality=1)

    # 检查解是否为魔方
    x = np.round(res.x)
    s = (numbers.flatten() * x).reshape(n**2, n, n)
    square = np.sum(s, axis=0)
    np.testing.assert_allclose(square.sum(axis=0), M)
    np.testing.assert_allclose(square.sum(axis=1), M)
    np.testing.assert_allclose(np.diag(square).sum(), M)
    np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)


def test_milp_2():
    # 解决带不等式约束和所有整数约束的 MIP
    # 参考资料：slide 5, https://www.cs.upc.edu/~erodri/webpage/cps/theory/lp/milp/slides.pdf
    # 同时检查 milp 函数接受所有有效的约束规范方式
    c = -np.ones(2)
    A = [[-2, 2], [-8, 10]]
    b_l = [1, -np.inf]
    b_u = [np.inf, 13]
    linear_constraint = LinearConstraint(A, b_l, b_u)

    # 解决原始问题
    res1 = milp(c=c, constraints=(A, b_l, b_u), integrality=True)
    res2 = milp(c=c, constraints=linear_constraint, integrality=True)
    res3 = milp(c=c, constraints=[(A, b_l, b_u)], integrality=True)
    # 调用 MILP 求解器，解决优化问题，要求整数解
    res4 = milp(c=c, constraints=[linear_constraint], integrality=True)

    # 调用 MILP 求解器，解决优化问题，要求整数解，并且指定多个线性约束
    res5 = milp(c=c, integrality=True,
                constraints=[(A[:1], b_l[:1], b_u[:1]),
                             (A[1:], b_l[1:], b_u[1:])])

    # 调用 MILP 求解器，解决优化问题，要求整数解，并且指定多个线性约束对象
    res6 = milp(c=c, integrality=True,
                constraints=[LinearConstraint(A[:1], b_l[:1], b_u[:1]),
                             LinearConstraint(A[1:], b_l[1:], b_u[1:])])

    # 调用 MILP 求解器，解决优化问题，要求整数解，并且混合使用元组和线性约束对象指定约束
    res7 = milp(c=c, integrality=True,
                constraints=[(A[:1], b_l[:1], b_u[:1]),
                             LinearConstraint(A[1:], b_l[1:], b_u[1:])])

    # 将多个求解结果的变量 x 组合成数组 xs
    xs = np.array([res1.x, res2.x, res3.x, res4.x, res5.x, res6.x, res7.x])

    # 将多个求解结果的目标函数值组合成数组 funs
    funs = np.array([res1.fun, res2.fun, res3.fun,
                     res4.fun, res5.fun, res6.fun, res7.fun])

    # 使用 np.testing.assert_allclose 函数检查 xs 是否与 [1, 2] 的广播形式一致
    np.testing.assert_allclose(xs, np.broadcast_to([1, 2], xs.shape))

    # 使用 np.testing.assert_allclose 函数检查 funs 是否全部接近于 -3
    np.testing.assert_allclose(funs, -3)

    # 解决松弛问题，不要求整数解，仅使用线性约束 (A, b_l, b_u)
    res = milp(c=c, constraints=(A, b_l, b_u))

    # 使用 np.testing.assert_allclose 函数检查 res.x 是否接近于 [4, 4.5]
    np.testing.assert_allclose(res.x, [4, 4.5])

    # 使用 np.testing.assert_allclose 函数检查 res.fun 是否接近于 -8.5
    np.testing.assert_allclose(res.fun, -8.5)
def test_milp_3():
    # 解决具有不等式约束和所有整数约束的 MIP 问题
    # 来源：https://en.wikipedia.org/wiki/Integer_programming#Example
    c = [0, -1]  # 目标函数的系数向量
    A = [[-1, 1], [3, 2], [2, 3]]  # 不等式约束的系数矩阵
    b_u = [1, 12, 12]  # 不等式约束的上界
    b_l = np.full_like(b_u, -np.inf, dtype=np.float64)  # 不等式约束的下界
    constraints = LinearConstraint(A, b_l, b_u)  # 创建线性约束对象

    integrality = np.ones_like(c)  # 标识所有变量为整数

    # 解决原始问题
    res = milp(c=c, constraints=constraints, integrality=integrality)  # 调用求解器求解问题
    assert_allclose(res.fun, -2)  # 断言目标函数值接近 -2
    # 可能有两个最优解，只需要其中一个
    assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])

    # 解决放宽后的问题
    res = milp(c=c, constraints=constraints)  # 调用求解器求解放宽后的问题
    assert_allclose(res.fun, -2.8)  # 断言目标函数值接近 -2.8
    assert_allclose(res.x, [1.8, 2.8])


def test_milp_4():
    # 解决具有不等式约束和仅一个整数约束的 MIP 问题
    # 来源：https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [8, 1]  # 目标函数的系数向量
    integrality = [0, 1]  # 第二个变量是整数
    A = [[1, 2], [-4, -1], [2, 1]]  # 不等式约束的系数矩阵
    b_l = [-14, -np.inf, -np.inf]  # 不等式约束的下界
    b_u = [np.inf, -33, 20]  # 不等式约束的上界
    constraints = LinearConstraint(A, b_l, b_u)  # 创建线性约束对象
    bounds = Bounds(-np.inf, np.inf)  # 变量的界限

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)  # 调用求解器求解问题
    assert_allclose(res.fun, 59)  # 断言目标函数值接近 59
    assert_allclose(res.x, [6.5, 7])


def test_milp_5():
    # 解决具有不等式和等式约束的 MIP 问题
    # 来源：https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [-3, -2, -1]  # 目标函数的系数向量
    integrality = [0, 0, 1]  # 第三个变量是整数
    lb = [0, 0, 0]  # 变量的下界
    ub = [np.inf, np.inf, 1]  # 变量的上界
    bounds = Bounds(lb, ub)  # 变量的界限对象
    A = [[1, 1, 1], [4, 2, 1]]  # 约束的系数矩阵
    b_l = [-np.inf, 12]  # 约束的下界
    b_u = [7, 12]  # 约束的上界
    constraints = LinearConstraint(A, b_l, b_u)  # 创建线性约束对象

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)  # 调用求解器求解问题
    # 存在多个解
    assert_allclose(res.fun, -12)


@pytest.mark.slow
@pytest.mark.timeout(120)  # prerelease_deps_coverage_64bit_blas job
def test_milp_6():
    # 解决具有仅等式约束的较大规模 MIP 问题
    # 来源：https://www.mathworks.com/help/optim/ug/intlinprog.html
    integrality = 1  # 所有变量为整数
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],
                     [39, 16, 22, 28, 26, 30, 23, 24],
                     [18, 14, 29, 27, 30, 38, 26, 26],
                     [41, 26, 28, 36, 18, 38, 16, 26]])  # 等式约束的系数矩阵
    b_eq = np.array([7872, 10466, 11322, 12058])  # 等式约束的右侧常数向量
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])  # 目标函数的系数向量

    res = milp(c=c, constraints=(A_eq, b_eq, b_eq), integrality=integrality)  # 调用求解器求解问题

    np.testing.assert_allclose(res.fun, 1854)  # 断言目标函数值接近 1854


def test_infeasible_prob_16609():
    # 确保预处理不会将明显不可行的问题标记为最优解 -- 参见 gh-16609
    c = [1.0, 0.0]  # 目标函数的系数向量
    integrality = [0, 1]  # 第二个变量是整数

    lb = [0, -np.inf]  # 变量的下界
    ub = [np.inf, np.inf]  # 变量的上界
    bounds = Bounds(lb, ub)  # 变量的界限对象

    A_eq = [[0.0, 1.0]]  # 等式约束的系数矩阵
    b_eq = [0.5]  # 等式约束的右侧常数向量
    constraints = LinearConstraint(A_eq, b_eq, b_eq)  # 创建线性约束对象
    # 调用名为 milp 的函数，进行混合整数线性规划计算，并将结果赋给 res 变量
    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    # 使用 NumPy 的测试工具 np.testing 来断言 res 的状态为 2
    np.testing.assert_equal(res.status, 2)
# 定义用于超时情况的消息字符串，标记时间限制和迭代限制
_msg_time = "Time limit reached. (HiGHS Status 13:"
_msg_iter = "Iteration limit reached. (HiGHS Status 14:"

# 标记此测试函数，如果运行环境不支持64位整型指针，则跳过执行，理由是未处理的32位GCC浮点数bug
@pytest.mark.skipif(np.intp(0).itemsize < 8,
                    reason="Unhandled 32-bit GCC FP bug")
# 标记此测试函数为慢速测试
@pytest.mark.slow
# 使用参数化测试，分别传入时间限制和迭代限制的选项及相关消息
@pytest.mark.parametrize(["options", "msg"], [({"time_limit": 0.1}, _msg_time),
                                              ({"node_limit": 1}, _msg_iter)])
def test_milp_timeout_16545(options, msg):
    # 确保如果 MILP 求解器超时，则不会丢弃解决方案
    # -- 参见 issue gh-16545

    # 初始化随机数生成器
    rng = np.random.default_rng(5123833489170494244)
    # 生成随机整数矩阵 A，大小为 100x100，值域在 0 到 5 之间
    A = rng.integers(0, 5, size=(100, 100))
    # 构造下界向量 b_lb，全为负无穷
    b_lb = np.full(100, fill_value=-np.inf)
    # 构造上界向量 b_ub，全为 25
    b_ub = np.full(100, fill_value=25)
    # 创建线性约束对象，限制条件为 A*x 在 b_lb 和 b_ub 之间
    constraints = LinearConstraint(A, b_lb, b_ub)
    # 初始化变量下界为全零，上界为全一
    variable_lb = np.zeros(100)
    variable_ub = np.ones(100)
    # 创建变量边界对象，限制变量 x 在 variable_lb 和 variable_ub 之间
    variable_bounds = Bounds(variable_lb, variable_ub)
    # 定义整数规划的整数变量
    integrality = np.ones(100)
    # 定义线性目标函数向量 c_vector，全为负一
    c_vector = -np.ones(100)
    # 调用 MILP 求解函数 milp，求解整数线性规划问题
    res = milp(
        c_vector,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options,
    )

    # 断言求解器的消息以给定的 msg 开头
    assert res.message.startswith(msg)
    # 断言返回结果中 x 不为 None
    assert res["x"] is not None

    # 确保解是可行的
    x = res["x"]
    tol = 1e-8  # 由于有限数值精度，有时需要容差
    assert np.all(b_lb - tol <= A @ x) and np.all(A @ x <= b_ub + tol)
    assert np.all(variable_lb - tol <= x) and np.all(x <= variable_ub + tol)
    assert np.allclose(x, np.round(x))


def test_three_constraints_16878():
    # `milp` 在传递恰好三个约束时失败的问题
    # 确保这种情况不再发生

    # 初始化随机数生成器
    rng = np.random.default_rng(5123833489170494244)
    # 生成随机整数矩阵 A，大小为 6x6，值域在 0 到 5 之间
    A = rng.integers(0, 5, size=(6, 6))
    # 构造下界向量 bl，全为负无穷
    bl = np.full(6, fill_value=-np.inf)
    # 构造上界向量 bu，全为 10
    bu = np.full(6, fill_value=10)
    # 创建线性约束列表，分别对应三组约束
    constraints = [LinearConstraint(A[:2], bl[:2], bu[:2]),
                   LinearConstraint(A[2:4], bl[2:4], bu[2:4]),
                   LinearConstraint(A[4:], bl[4:], bu[4:])]
    # 创建另一种线性约束组合形式，与上述保持一致
    constraints2 = [(A[:2], bl[:2], bu[:2]),
                    (A[2:4], bl[2:4], bu[2:4]),
                    (A[4:], bl[4:], bu[4:])]
    # 创建变量下界 lb 全为零，上界 ub 全为一
    lb = np.zeros(6)
    ub = np.ones(6)
    # 创建变量边界对象，限制变量在 lb 和 ub 之间
    variable_bounds = Bounds(lb, ub)
    # 定义线性目标函数向量 c，全为负一
    c = -np.ones(6)
    # 分别调用 MILP 求解函数 milp，传入不同的约束形式，进行求解
    res1 = milp(c, bounds=variable_bounds, constraints=constraints)
    res2 = milp(c, bounds=variable_bounds, constraints=constraints2)
    ref = milp(c, bounds=variable_bounds, constraints=(A, bl, bu))
    # 断言所有求解成功
    assert res1.success and res2.success
    # 断言所有结果 x 接近参考结果的 x
    assert_allclose(res1.x, ref.x)
    assert_allclose(res2.x, ref.x)


@pytest.mark.xslow
def test_mip_rel_gap_passdown():
    # 解决问题，逐渐减小 mip_gap，以确保 mip_rel_gap 减小
    # 改编自 test_linprog::TestLinprogHiGHSMIP::test_mip_rel_gap_passdown
    # MIP 取自 test_mip_6

    # 定义等式约束矩阵 A_eq
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],
                     [39, 16, 22, 28, 26, 30, 23, 24],
                     [18, 14, 29, 27, 30, 38, 26, 26],
                     [41, 26, 28, 36, 18, 38, 16, 26]])
    # 定义一个包含整数的 NumPy 数组，表示线性规划中的等式约束的右侧值
    b_eq = np.array([7872, 10466, 11322, 12058])

    # 定义一个包含整数的 NumPy 数组，表示线性规划中的目标函数系数
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])

    # 定义一组浮点数，表示需要测试的 MIP 相对间隙（相对最优性间隙）
    mip_rel_gaps = [0.25, 0.01, 0.001]

    # 用于存储每次 MIP 求解后得到的实际相对间隙值的空列表
    sol_mip_gaps = []

    # 对于每个 mip_rel_gap 进行迭代
    for mip_rel_gap in mip_rel_gaps:
        # 调用 milp 函数进行混合整数线性规划求解，返回结果存储在 res 中
        res = milp(c=c, bounds=(0, np.inf), constraints=(A_eq, b_eq, b_eq),
                   integrality=True, options={"mip_rel_gap": mip_rel_gap})

        # 断言求解的实际相对间隙小于等于给定的 mip_rel_gap
        assert res.mip_gap <= mip_rel_gap

        # 检查 res.mip_gap 是否满足文档中的定义
        assert res.mip_gap == (res.fun - res.mip_dual_bound)/res.fun

        # 将求解得到的实际相对间隙值添加到 sol_mip_gaps 列表中
        sol_mip_gaps.append(res.mip_gap)

    # 确保 mip_rel_gap 参数实际起作用
    # 检查不同 mip_rel_gap 参数下求解得到的间隙值差异是否单调下降
    assert np.all(np.diff(sol_mip_gaps) < 0)
```