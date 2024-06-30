# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_linprog.py`

```
"""
Unit test for Linear Programming
"""
# 导入系统相关模块
import sys
# 导入平台信息模块
import platform

# 导入 NumPy 库及其测试相关模块
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_less, assert_warns, suppress_warnings)
# 导入 pytest 中的断言模块
from pytest import raises as assert_raises

# 导入 SciPy 中的线性规划模块及相关警告
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest

# 检查是否导入了 Umfpack 警告模块
has_umfpack = True
try:
    from scikits.umfpack import UmfpackWarning
except ImportError:
    has_umfpack = False

# 检查是否导入了 Cholmod 相关模块
has_cholmod = True
try:
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod  # noqa: F401
except ImportError:
    has_cholmod = False


def _assert_iteration_limit_reached(res, maxiter):
    # 断言函数：检查是否未正确报告成功，以及是否正确报告了迭代次数
    assert_(not res.success, "Incorrectly reported success")
    assert_(res.success < maxiter, "Incorrectly reported number of iterations")
    assert_equal(res.status, 1, "Failed to report iteration limit reached")


def _assert_infeasible(res):
    # 断言函数：检查是否未正确报告成功，并且是否正确报告了不可行状态
    # res: 线性规划结果对象
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report infeasible status")


def _assert_unbounded(res):
    # 断言函数：检查是否未正确报告成功，并且是否正确报告了无界状态
    # res: 线性规划结果对象
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 3, "failed to report unbounded status")


def _assert_unable_to_find_basic_feasible_sol(res):
    # 断言函数：检查是否未正确报告成功，并且根据情况报告了找不到基本可行解的状态
    # res: 线性规划结果对象
    # 状态可能是2或4，具体取决于无法找到基本可行解的原因
    # 如果预期问题不具有可行解，则应使用 _assert_infeasible
    assert_(not res.success, "incorrectly reported success")
    assert_(res.status in (2, 4), "failed to report optimization failure")


def _assert_success(res, desired_fun=None, desired_x=None,
                    rtol=1e-8, atol=1e-8):
    # 断言函数：检查是否成功运行线性规划，并且检查目标函数值和解是否符合预期
    # res: 线性规划结果对象
    # desired_fun: 预期的目标函数值或 None
    # desired_x: 预期的解或 None
    if not res.success:
        msg = f"linprog status {res.status}, message: {res.message}"
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun,
                        err_msg="converged to an unexpected objective value",
                        rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)


def magic_square(n):
    """
    Generates a linear program for which integer solutions represent an
    n x n magic square; binary decision variables represent the presence
    (or absence) of an integer 1 to n^2 in each position of the square.
    """
    # 生成一个线性规划问题，其中整数解表示一个 n x n 的幻方
    # 二进制决策变量表示在方格的每个位置上是否存在 1 到 n^2 的整数
    # 设置随机数种子为0，保证结果的可重复性
    np.random.seed(0)
    # 计算矩阵的期望和M
    M = n * (n**2 + 1) / 2

    # 生成从1到n^2的连续整数数组
    numbers = np.arange(n**4) // n**2 + 1

    # 将一维数组转换为n×n×n的三维数组
    numbers = numbers.reshape(n**2, n, n)

    # 创建一个全零的n×n×n的三维数组
    zeros = np.zeros((n**2, n, n))

    # 初始化 A_list 和 b_list 用于存储约束条件的系数和右侧常数
    A_list = []
    b_list = []

    # Rule 1: use every number exactly once
    # 每个数字恰好使用一次的约束条件
    for i in range(n**2):
        # 创建一个与 zeros 大小相同的副本数组，并在第i个位置设置为1
        A_row = zeros.copy()
        A_row[i, :, :] = 1
        A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
        b_list.append(1)  # 添加右侧常数1到 b_list

    # Rule 2: Only one number per square
    # 每个小方块中只能有一个数字的约束条件
    for i in range(n):
        for j in range(n):
            # 创建一个与 zeros 大小相同的副本数组，并在第i行第j列的所有n个平面上设置为1
            A_row = zeros.copy()
            A_row[:, i, j] = 1
            A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
            b_list.append(1)  # 添加右侧常数1到 b_list

    # Rule 3: sum of rows is M
    # 每行数字之和等于M的约束条件
    for i in range(n):
        # 创建一个与 zeros 大小相同的副本数组，并在第i列的所有平面上设置为对应的 numbers 数组值
        A_row = zeros.copy()
        A_row[:, i, :] = numbers[:, i, :]
        A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
        b_list.append(M)  # 添加右侧常数M到 b_list

    # Rule 4: sum of columns is M
    # 每列数字之和等于M的约束条件
    for i in range(n):
        # 创建一个与 zeros 大小相同的副本数组，并在第i个平面的所有行上设置为对应的 numbers 数组值
        A_row = zeros.copy()
        A_row[:, :, i] = numbers[:, :, i]
        A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
        b_list.append(M)  # 添加右侧常数M到 b_list

    # Rule 5: sum of diagonals is M
    # 对角线上数字之和等于M的约束条件
    A_row = zeros.copy()
    A_row[:, range(n), range(n)] = numbers[:, range(n), range(n)]
    A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
    b_list.append(M)  # 添加右侧常数M到 b_list
    A_row = zeros.copy()
    A_row[:, range(n), range(-1, -n - 1, -1)] = \
        numbers[:, range(n), range(-1, -n - 1, -1)]
    A_list.append(A_row.flatten())  # 将二维数组扁平化后添加到 A_list
    b_list.append(M)  # 添加右侧常数M到 b_list

    # 将 A_list 转换为 numpy 数组 A，并设置数据类型为 float
    A = np.array(np.vstack(A_list), dtype=float)
    # 将 b_list 转换为 numpy 数组 b，并设置数据类型为 float
    b = np.array(b_list, dtype=float)
    # 创建一个随机数组 c，长度与 A 的列数相同
    c = np.random.rand(A.shape[1])

    # 返回计算结果 A, b, c，以及原始的 numbers 数组和 M 值
    return A, b, c, numbers, M
def lpgen_2d(m, n):
    """ -> A b c LP test: m*n vars, m+n constraints
        row sums == n/m, col sums == 1
        https://gist.github.com/denis-bz/8647461
    """
    # 设置随机种子为0，以保证可复现性
    np.random.seed(0)
    # 生成大小为(m, n)的指数分布的负数数组作为目标函数系数
    c = - np.random.exponential(size=(m, n))
    
    # 初始化Arow矩阵和brow向量
    Arow = np.zeros((m, m * n))
    brow = np.zeros(m)
    # 为Arow和brow赋值，构建m个约束条件，每个约束条件中包含n个变量
    for j in range(m):
        j1 = j + 1
        Arow[j, j * n:j1 * n] = 1
        brow[j] = n / m

    # 初始化Acol矩阵和bcol向量
    Acol = np.zeros((n, m * n))
    bcol = np.zeros(n)
    # 为Acol和bcol赋值，构建n个约束条件，每个约束条件中包含m个变量
    for j in range(n):
        j1 = j + 1
        Acol[j, j::n] = 1
        bcol[j] = 1

    # 将Arow和Acol垂直堆叠形成A矩阵，将brow和bcol水平堆叠形成b向量
    A = np.vstack((Arow, Acol))
    b = np.hstack((brow, bcol))

    # 返回A矩阵、b向量和展平后的目标函数系数c
    return A, b, c.ravel()


def very_random_gen(seed=0):
    # 设置随机种子
    np.random.seed(seed)
    # 初始化参数
    m_eq, m_ub, n = 10, 20, 50
    # 生成大小为n的随机浮点数数组作为目标函数系数c
    c = np.random.rand(n)-0.5
    # 生成大小为(m_ub, n)的随机浮点数数组作为A_ub矩阵
    A_ub = np.random.rand(m_ub, n)-0.5
    # 生成大小为m_ub的随机浮点数数组作为b_ub向量
    b_ub = np.random.rand(m_ub)-0.5
    # 生成大小为(m_eq, n)的随机浮点数数组作为A_eq矩阵
    A_eq = np.random.rand(m_eq, n)-0.5
    # 生成大小为m_eq的随机浮点数数组作为b_eq向量
    b_eq = np.random.rand(m_eq)-0.5
    # 生成大小为n的随机浮点数数组作为下界lb
    lb = -np.random.rand(n)
    # 生成大小为n的随机浮点数数组作为上界ub
    ub = np.random.rand(n)
    # 根据条件将lb中小于随机浮点数的元素设为负无穷
    lb[lb < -np.random.rand()] = -np.inf
    # 根据条件将ub中大于随机浮点数的元素设为正无穷
    ub[ub > np.random.rand()] = np.inf
    # 将lb和ub堆叠形成bounds矩阵
    bounds = np.vstack((lb, ub)).T
    # 返回c, A_ub, b_ub, A_eq, b_eq, bounds
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def nontrivial_problem():
    # 定义目标函数系数c
    c = [-1, 8, 4, -6]
    # 定义不等式约束矩阵A_ub
    A_ub = [[-7, -7, 6, 9],
            [1, -1, -3, 0],
            [10, -10, -7, 7],
            [6, -1, 3, 4]]
    # 定义不等式约束向量b_ub
    b_ub = [-3, 6, -6, 6]
    # 定义等式约束矩阵A_eq
    A_eq = [[-10, 1, 1, -8]]
    # 定义等式约束向量b_eq
    b_eq = [-4]
    # 定义最优解向量x_star
    x_star = [101 / 1391, 1462 / 1391, 0, 752 / 1391]
    # 定义最优解值f_star
    f_star = 7083 / 1391
    # 返回c, A_ub, b_ub, A_eq, b_eq, x_star, f_star
    return c, A_ub, b_ub, A_eq, b_eq, x_star, f_star


def l1_regression_prob(seed=0, m=8, d=9, n=100):
    '''
    Training data is {(x0, y0), (x1, y2), ..., (xn-1, yn-1)}
        x in R^d
        y in R
    n: number of training samples
    d: dimension of x, i.e. x in R^d
    phi: feature map R^d -> R^m
    m: dimension of feature space
    '''
    # 设置随机种子
    np.random.seed(seed)
    # 生成大小为(m, d)的正态分布随机数数组作为特征映射phi
    phi = np.random.normal(0, 1, size=(m, d))  # random feature mapping
    # 生成大小为m的正态分布随机数数组作为真实权重向量w_true
    w_true = np.random.randn(m)
    # 生成大小为(d, n)的正态分布随机数数组作为特征x
    x = np.random.normal(0, 1, size=(d, n))  # features
    # 计算y值，包括真实权重w_true对特征映射phi @ x的影响和加上小量噪声的影响
    y = w_true @ (phi @ x) + np.random.normal(0, 1e-5, size=n)  # measurements

    # 构建问题
    # 初始化目标函数系数c
    c = np.ones(m+n)
    c[:m] = 0
    # 初始化不等式约束矩阵A_ub为稀疏矩阵
    A_ub = scipy.sparse.lil_matrix((2*n, n+m))
    idx = 0
    # 为A_ub赋值，构建2n个约束条件
    for ii in range(n):
        A_ub[idx, :m] = phi @ x[:, ii]
        A_ub[idx, m+ii] = -1
        A_ub[idx+1, :m] = -1*phi @ x[:, ii]
        A_ub[idx+1, m+ii] = -1
        idx += 2
    A_ub = A_ub.tocsc()  # 转换为压缩列格式
    # 初始化不等式约束向量b_ub
    b_ub = np.zeros(2*n)
    b_ub[0::2] = y
    b_ub[1::2] = -y
    # 定义变量的界限bnds
    bnds = [(None, None)]*m + [(0, None)]*n
    # 返回c, A_ub, b_ub, bnds
    return c, A_ub, b_ub, bnds


def generic_callback_test(self):
    # 检查回调函数是否按预期工作
    last_cb = {}
    # 定义回调函数cb，接受一个参数res，表示优化器的返回结果
    def cb(res):
        # 从返回结果中弹出'message'键对应的值，并赋给变量message
        message = res.pop('message')
        # 从返回结果中弹出'complete'键对应的值，并赋给变量complete
        complete = res.pop('complete')

        # 使用断言确保'res'字典中'phase'键对应的值是1或2
        assert_(res.pop('phase') in (1, 2))
        # 使用断言确保'res'字典中'status'键对应的值在0到3之间（包含0和3）
        assert_(res.pop('status') in range(4))
        # 使用断言确保'res'字典中'nit'键对应的值是整数类型
        assert_(isinstance(res.pop('nit'), int))
        # 使用断言确保'complete'变量的值是布尔类型
        assert_(isinstance(complete, bool))
        # 使用断言确保'message'变量的值是字符串类型
        assert_(isinstance(message, str))

        # 将'res'字典中的'x'键对应的值赋给last_cb字典的'x'键
        last_cb['x'] = res['x']
        # 将'res'字典中的'fun'键对应的值赋给last_cb字典的'fun'键
        last_cb['fun'] = res['fun']
        # 将'res'字典中的'slack'键对应的值赋给last_cb字典的'slack'键
        last_cb['slack'] = res['slack']
        # 将'res'字典中的'con'键对应的值赋给last_cb字典的'con'键

    # 创建一个包含两个元素的NumPy数组，用于线性规划的目标函数系数
    c = np.array([-3, -2])
    # 创建一个二维列表，用于线性规划中的不等式约束的系数矩阵A_ub
    A_ub = [[2, 1], [1, 1], [1, 0]]
    # 创建一个一维列表，用于线性规划中的不等式约束的右侧常数向量b_ub
    b_ub = [10, 8, 4]
    # 调用线性规划函数linprog，传入目标函数系数c、不等式约束系数矩阵A_ub、不等式约束右侧常数向量b_ub，
    # 以及回调函数cb和方法参数self.method，并将结果赋给变量res
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)

    # 调用自定义的断言函数_assert_success，验证优化结果是否符合预期
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])
    # 使用assert_allclose函数检查last_cb字典中'fun'键对应的值与res字典中'fun'键对应的值的近似程度
    assert_allclose(last_cb['fun'], res['fun'])
    # 使用assert_allclose函数检查last_cb字典中'x'键对应的值与res字典中'x'键对应的值的近似程度
    assert_allclose(last_cb['x'], res['x'])
    # 使用assert_allclose函数检查last_cb字典中'con'键对应的值与res字典中'con'键对应的值的近似程度
    assert_allclose(last_cb['con'], res['con'])
    # 使用assert_allclose函数检查last_cb字典中'slack'键对应的值与res字典中'slack'键对应的值的近似程度
    assert_allclose(last_cb['slack'], res['slack'])
def test_unknown_solvers_and_options():
    # 定义线性规划问题的参数
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    # 测试：期望引发 ValueError 异常，因为使用了未知的求解方法 'ekki-ekki-ekki'
    assert_raises(ValueError, linprog,
                  c, A_ub=A_ub, b_ub=b_ub, method='ekki-ekki-ekki')
    
    # 测试：期望引发 ValueError 异常，因为使用了未知的求解方法 'highs-ekki'
    assert_raises(ValueError, linprog,
                  c, A_ub=A_ub, b_ub=b_ub, method='highs-ekki')
    
    # 测试：期望引发 OptimizeWarning 警告，因为使用了未知的选项 'rr_method'
    message = "Unrecognized options detected: {'rr_method': 'ekki-ekki-ekki'}"
    with pytest.warns(OptimizeWarning, match=message):
        linprog(c, A_ub=A_ub, b_ub=b_ub,
                options={"rr_method": 'ekki-ekki-ekki'})


def test_choose_solver():
    # 'highs' 方法选择 'dual' 求解器
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    # 测试：使用 'highs' 方法求解线性规划问题
    res = linprog(c, A_ub, b_ub, method='highs')
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])


def test_deprecation():
    # 测试：使用过时的求解方法 'interior-point'，应引发 DeprecationWarning 警告
    with pytest.warns(DeprecationWarning):
        linprog(1, method='interior-point')
    # 测试：使用过时的求解方法 'revised simplex'，应引发 DeprecationWarning 警告
    with pytest.warns(DeprecationWarning):
        linprog(1, method='revised simplex')
    # 测试：使用过时的求解方法 'simplex'，应引发 DeprecationWarning 警告
    with pytest.warns(DeprecationWarning):
        linprog(1, method='simplex')


def test_highs_status_message():
    # 测试 'highs' 求解器的不同状态消息
    
    # 测试：成功优化，期望状态为 0
    res = linprog(1, method='highs')
    msg = "Optimization terminated successfully. (HiGHS Status 7:"
    assert res.status == 0
    assert res.message.startswith(msg)

    # 测试：超时限制，期望状态为 1
    A, b, c, numbers, M = magic_square(6)
    bounds = [(0, 1)] * len(c)
    integrality = [1] * len(c)
    options = {"time_limit": 0.1}
    res = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds, method='highs',
                  options=options, integrality=integrality)
    msg = "Time limit reached. (HiGHS Status 13:"
    assert res.status == 1
    assert res.message.startswith(msg)

    # 测试：达到迭代次数限制，期望状态为 1
    options = {"maxiter": 10}
    res = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds, method='highs-ds',
                  options=options)
    msg = "Iteration limit reached. (HiGHS Status 14:"
    assert res.status == 1
    assert res.message.startswith(msg)

    # 测试：问题不可行，期望状态为 2
    res = linprog(1, bounds=(1, -1), method='highs')
    msg = "The problem is infeasible. (HiGHS Status 8:"
    assert res.status == 2
    assert res.message.startswith(msg)

    # 测试：问题无界，期望状态为 3
    res = linprog(-1, method='highs')
    msg = "The problem is unbounded. (HiGHS Status 10:"
    assert res.status == 3
    assert res.message.startswith(msg)

    # 测试：未识别的 HiGHS 状态码，期望状态为 4
    from scipy.optimize._linprog_highs import _highs_to_scipy_status_message
    status, message = _highs_to_scipy_status_message(58, "Hello!")
    msg = "The HiGHS status code was not recognized. (HiGHS Status 58:"
    assert status == 4
    assert message.startswith(msg)

    # 测试：未提供 HiGHS 状态码，期望状态为 4
    status, message = _highs_to_scipy_status_message(None, None)
    msg = "HiGHS did not provide a status code. (HiGHS Status None: None)"
    assert status == 4
    assert message.startswith(msg)


def test_bug_17380():
    # 测试修复 bug 17380
    linprog([1, 1], A_ub=[[-1, 0]], b_ub=[-2.5], integrality=[1, 1])

# 初始化变量为 None
A_ub = None
b_ub = None
A_eq = None
b_eq = None
bounds = None

################
# Common Tests #
################
    """
    Base class for `linprog` tests. Generally, each test will be performed
    once for every derived class of LinprogCommonTests, each of which will
    typically change self.options and/or self.method. Effectively, these tests
    are run for many combination of method (simplex, revised simplex, and
    interior point) and options (such as pivoting rule or sparse treatment).
    """

    ##################
    # Targeted Tests #
    ##################

    # 测试回调函数是否正常工作
    def test_callback(self):
        generic_callback_test(self)

    # 测试显示选项不会导致任何问题
    def test_disp(self):
        # 生成一个简单的线性规划问题
        A, b, c = lpgen_2d(20, 20)
        # 运行线性规划求解，并设置显示选项为 True
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"disp": True})
        # 断言求解成功，并验证期望的目标函数值
        _assert_success(res, desired_fun=-64.049494229)

    # 测试文档字符串中的示例
    def test_docstring_example(self):
        # 定义线性规划问题的参数
        c = [-1, 4]
        A = [[-3, 1], [1, 2]]
        b = [6, 4]
        x0_bounds = (None, None)
        x1_bounds = (-3, None)
        # 运行线性规划求解，并使用指定的方法和选项
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
                      options=self.options, method=self.method)
        # 断言求解成功，并验证期望的目标函数值
        _assert_success(res, desired_fun=-22)

    # 测试类型错误是否被正确处理
    def test_type_error(self):
        # 定义一个可能引发类型错误的线性规划问题
        c = [1]
        A_eq = [[1]]
        b_eq = "hello"
        # 断言在传入类型错误时会引发 TypeError 异常
        assert_raises(TypeError, linprog,
                      c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)

    # 测试 b_ub 别名问题
    def test_aliasing_b_ub(self):
        # 定义一个简单的线性规划问题和边界
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub_orig = np.array([3.0])
        b_ub = b_ub_orig.copy()
        bounds = (-4.0, np.inf)
        # 运行线性规划求解，并验证不会修改 b_ub
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解成功，并验证期望的目标函数值和解向量
        _assert_success(res, desired_fun=-4, desired_x=[-4])
        # 断言 b_ub 原始值没有被修改
        assert_allclose(b_ub_orig, b_ub)

    # 测试 b_eq 别名问题
    def test_aliasing_b_eq(self):
        # 定义一个简单的线性规划问题和边界
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq_orig = np.array([3.0])
        b_eq = b_eq_orig.copy()
        bounds = (-4.0, np.inf)
        # 运行线性规划求解，并验证不会修改 b_eq
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解成功，并验证期望的目标函数值和解向量
        _assert_success(res, desired_fun=3, desired_x=[3])
        # 断言 b_eq 原始值没有被修改
        assert_allclose(b_eq_orig, b_eq)
    def test_non_ndarray_args(self):
        # 检查 linprog 是否接受列表作为数组的替代品
        # 这个更详细地测试在 test__linprog_clean_inputs.py 中进行
        c = [1.0]  # 目标函数的系数
        A_ub = [[1.0]]  # 不等式约束的系数矩阵
        b_ub = [3.0]  # 不等式约束的右侧常数
        A_eq = [[1.0]]  # 等式约束的系数矩阵
        b_eq = [2.0]  # 等式约束的右侧常数
        bounds = (-1.0, 10.0)  # 变量的上下界
        # 调用 linprog 函数，传入参数并指定优化方法和选项
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言优化成功，并检查期望的目标函数值和变量值
        _assert_success(res, desired_fun=2, desired_x=[2])

    def test_unknown_options(self):
        c = np.array([-3, -2])  # 目标函数的系数向量
        A_ub = [[2, 1], [1, 1], [1, 0]]  # 不等式约束的系数矩阵
        b_ub = [10, 8, 4]  # 不等式约束的右侧常数向量

        def f(c, A_ub=None, b_ub=None, A_eq=None,
              b_eq=None, bounds=None, options={}):
            # 调用 linprog 函数，传入参数并指定优化方法和选项
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=options)

        # 复制现有选项字典并添加新的选项 'spam'
        o = {key: self.options[key] for key in self.options}
        o['spam'] = 42

        # 断言调用函数 f 时会产生 OptimizeWarning 警告
        assert_warns(OptimizeWarning, f,
                     c, A_ub=A_ub, b_ub=b_ub, options=o)

    def test_integrality_without_highs(self):
        # 确保在没有使用 method='highs' 的情况下使用 'integrality' 参数
        # 会引发警告并产生放宽问题的正确解决方案
        # 参考：https://en.wikipedia.org/wiki/Integer_programming#Example
        A_ub = np.array([[-1, 1], [3, 2], [2, 3]])  # 不等式约束的系数矩阵
        b_ub = np.array([1, 12, 12])  # 不等式约束的右侧常数向量
        c = -np.array([0, 1])  # 目标函数的系数向量

        bounds = [(0, np.inf)] * len(c)  # 变量的上下界列表
        integrality = [1] * len(c)  # 变量的整数性列表

        # 使用 assert_warns 来捕获 OptimizeWarning 警告
        with np.testing.assert_warns(OptimizeWarning):
            # 调用 linprog 函数，传入参数并指定优化方法和选项
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method=self.method, integrality=integrality)

        # 断言优化结果中变量值的近似相等性
        np.testing.assert_allclose(res.x, [1.8, 2.8])
        # 断言优化结果中目标函数值的近似相等性
        np.testing.assert_allclose(res.fun, -2.8)
    def test_invalid_inputs(self):
        # 定义内部函数 f，用于测试线性规划函数的不合法输入
        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            # 调用线性规划函数 linprog，传入参数和选项
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=self.options)

        # 测试不合格的边界条件
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4)])
        with np.testing.suppress_warnings() as sup:
            sup.filter(VisibleDeprecationWarning, "Creating an ndarray from ragged")
            assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4), (3, 4, 5)])
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, -2), (1, 2)])

        # 测试其他不合法的输入
        assert_raises(ValueError, f, [1, 2], A_ub=[[1, 2]], b_ub=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_ub=[[1]], b_ub=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1, 2]], b_eq=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1]], b_eq=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[1], b_eq=1)

        # 对于稀疏预处理，这个最后的检查对于 3-D 稀疏矩阵没有意义
        if ("_sparse_presolve" in self.options and
                self.options["_sparse_presolve"]):
            return
            # 不存在 3-D 稀疏矩阵

        assert_raises(ValueError, f, [1, 2], A_ub=np.zeros((1, 1, 3)), b_eq=1)

    def test_sparse_constraints(self):
        # gh-13559: 改进稀疏输入不支持时的错误消息
        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            # 调用线性规划函数 linprog，传入参数和选项
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=self.options)

        np.random.seed(0)
        m = 100
        n = 150
        A_eq = scipy.sparse.rand(m, n, 0.5)
        x_valid = np.random.randn(n)
        c = np.random.randn(n)
        ub = x_valid + np.random.rand(n)
        lb = x_valid - np.random.rand(n)
        bounds = np.column_stack((lb, ub))
        b_eq = A_eq * x_valid

        if self.method in {'simplex', 'revised simplex'}:
            # simplex 和 revised simplex 应该抛出错误
            with assert_raises(ValueError, match=f"Method '{self.method}' "
                               "does not support sparse constraint matrices."):
                linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                        method=self.method, options=self.options)
        else:
            # 其他方法应该成功运行
            options = {**self.options}
            if self.method in {'interior-point'}:
                options['sparse'] = True

            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                          method=self.method, options=options)
            assert res.success
    def test_bounds_fixed(self):
        # Test fixed bounds (upper equal to lower)

        # 获取是否进行预处理的选项，若未指定则默认为 True
        do_presolve = self.options.get('presolve', True)

        # 测试单变量情况下上下界相等的线性规划问题
        res = linprog([1], bounds=(1, 1),
                      method=self.method, options=self.options)
        # 验证求解成功且结果符合预期值和上下界
        _assert_success(res, 1, 1)
        # 如果进行了预处理，确认迭代次数为0
        if do_presolve:
            assert_equal(res.nit, 0)

        # 测试多变量情况下各变量的上下界相等的线性规划问题
        res = linprog([1, 2, 3], bounds=[(5, 5), (-1, -1), (3, 3)],
                      method=self.method, options=self.options)
        # 验证求解成功且结果符合预期值和上下界
        _assert_success(res, 12, [5, -1, 3])
        # 如果进行了预处理，确认迭代次数为0
        if do_presolve:
            assert_equal(res.nit, 0)

        # 测试多变量情况下部分变量上下界相等的线性规划问题
        res = linprog([1, 1], bounds=[(1, 1), (1, 3)],
                      method=self.method, options=self.options)
        # 验证求解成功且结果符合预期值和上下界
        _assert_success(res, 2, [1, 1])
        # 如果进行了预处理，确认迭代次数为0
        if do_presolve:
            assert_equal(res.nit, 0)

        # 测试带等式约束的多变量情况下部分变量上下界相等的线性规划问题
        res = linprog([1, 1, 2], A_eq=[[1, 0, 0], [0, 1, 0]], b_eq=[1, 7],
                      bounds=[(-5, 5), (0, 10), (3.5, 3.5)],
                      method=self.method, options=self.options)
        # 验证求解成功且结果符合预期值和上下界
        _assert_success(res, 15, [1, 7, 3.5])
        # 如果进行了预处理，确认迭代次数为0
        if do_presolve:
            assert_equal(res.nit, 0)
    def test_bounds_infeasible_2(self):
        # 测试不可行的边界条件 (下界为正无穷，上界为负无穷)
        # 如果启用了预处理选项，则测试预处理阶段是否找到解决方案（即迭代次数为0）。
        # 对于单纯形法，这些情况不会导致不可行状态，而是会产生 RuntimeWarning。
        # 这是 _presolve() 负责处理可行性检查的结果。参见问题 gh-11618。
        do_presolve = self.options.get('presolve', True)
        simplex_without_presolve = not do_presolve and self.method == 'simplex'

        # 定义测试中的目标函数系数
        c = [1, 2, 3]
        # 定义两组边界条件
        bounds_1 = [(1, 2), (np.inf, np.inf), (3, 4)]
        bounds_2 = [(1, 2), (-np.inf, -np.inf), (3, 4)]

        if simplex_without_presolve:
            # 定义测试函数 g，用于单纯形法测试
            def g(c, bounds):
                res = linprog(c, bounds=bounds,
                              method=self.method, options=self.options)
                return res

            # 断言预期的 RuntimeWarning
            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_1)

            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_2)
        else:
            # 在不使用 g 函数的情况下直接调用 linprog 进行测试
            res = linprog(c=c, bounds=bounds_1,
                          method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)
            
            res = linprog(c=c, bounds=bounds_2,
                          method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)

    def test_empty_constraint_1(self):
        # 定义目标函数系数
        c = [-1, -2]
        # 调用 linprog 函数进行测试
        res = linprog(c, method=self.method, options=self.options)
        # 断言预期的无界解
        _assert_unbounded(res)

    def test_empty_constraint_2(self):
        # 定义目标函数系数和边界条件
        c = [-1, 1, -1, 1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        # 调用 linprog 函数进行测试
        res = linprog(c, bounds=bounds,
                      method=self.method, options=self.options)
        # 断言预期的无界解
        _assert_unbounded(res)
        # 预处理阶段检测到无界解时，迭代次数为0
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_empty_constraint_3(self):
        # 定义目标函数系数和边界条件
        c = [1, -1, 1, -1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        # 调用 linprog 函数进行测试
        res = linprog(c, bounds=bounds,
                      method=self.method, options=self.options)
        # 断言预期的成功解，验证最优解和最优目标函数值
        _assert_success(res, desired_x=[0, 0, -1, 1], desired_fun=-2)
    def test_inequality_constraints(self):
        # Minimize linear function subject to linear inequality constraints.
        #  http://www.dam.brown.edu/people/huiwang/classes/am121/Archive/simplex_121_c.pdf
        # 定义要最小化的线性函数的系数向量，此处为最大化目标，因此取相反数
        c = np.array([3, 2]) * -1  # maximize

        # 定义不等式约束的系数矩阵
        A_ub = [[2, 1],
                [1, 1],
                [1, 0]]

        # 定义不等式约束的右侧向量
        b_ub = [10, 8, 4]

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的函数值和最优解
        _assert_success(res, desired_fun=-18, desired_x=[2, 6])

    def test_inequality_constraints2(self):
        # Minimize linear function subject to linear inequality constraints.
        # http://www.statslab.cam.ac.uk/~ff271/teaching/opt/notes/notes8.pdf
        # (dead link)
        # 定义要最小化的线性函数的系数向量
        c = [6, 3]

        # 定义不等式约束的系数矩阵
        A_ub = [[0, 3],
                [-1, -1],
                [-2, 1]]

        # 定义不等式约束的右侧向量
        b_ub = [2, -1, -1]

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的函数值和最优解
        _assert_success(res, desired_fun=5, desired_x=[2 / 3, 1 / 3])

    def test_bounds_simple(self):
        # 定义要最小化的线性函数的系数向量
        c = [1, 2]

        # 定义变量的上下界
        bounds = (1, 2)

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的最优解
        _assert_success(res, desired_x=[1, 1])

        # 修改变量的上下界为多个变量的情况
        bounds = [(1, 2), (1, 2)]

        # 再次调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的最优解
        _assert_success(res, desired_x=[1, 1])

    def test_bounded_below_only_1(self):
        # 定义要最小化的线性函数的系数向量
        c = np.array([1.0])

        # 定义等式约束的系数矩阵和右侧向量
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])

        # 定义变量的下界
        bounds = (1.0, None)

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的函数值和最优解
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_below_only_2(self):
        # 定义要最小化的线性函数的系数向量
        c = np.ones(3)

        # 定义等式约束的系数矩阵和右侧向量
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])

        # 定义变量的下界
        bounds = (0.5, np.inf)

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的最优解
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounded_above_only_1(self):
        # 定义要最小化的线性函数的系数向量
        c = np.array([1.0])

        # 定义等式约束的系数矩阵和右侧向量
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])

        # 定义变量的上界
        bounds = (None, 10.0)

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的函数值和最优解
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_above_only_2(self):
        # 定义要最小化的线性函数的系数向量
        c = np.ones(3)

        # 定义等式约束的系数矩阵和右侧向量
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])

        # 定义变量的上界
        bounds = (-np.inf, 4)

        # 调用线性规划函数，求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 对结果进行断言，验证是否达到期望的最优解
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))
    def test_bounds_infinity(self):
        # 创建一个包含三个全为1的数组作为目标函数系数向量
        c = np.ones(3)
        # 创建一个3x3的单位矩阵作为等式约束的系数矩阵
        A_eq = np.eye(3)
        # 创建一个长度为3的数组作为等式约束的右侧向量
        b_eq = np.array([1, 2, 3])
        # 定义一个变量的上下界为负无穷到正无穷
        bounds = (-np.inf, np.inf)
        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解结果为成功，期望的最优解为b_eq，期望的最优值为b_eq元素之和
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounds_mixed(self):
        # 定义一个数组作为目标函数系数向量，使其最大化
        c = np.array([-1, 4]) * -1  # maximize
        # 创建一个2x2的浮点数类型的不等式约束系数矩阵
        A_ub = np.array([[-3, 1],
                         [1, 2]], dtype=np.float64)
        # 创建一个长度为2的不等式约束右侧向量
        b_ub = [6, 4]
        # 定义两个变量的上下界
        x0_bounds = (-np.inf, np.inf)
        x1_bounds = (-3, np.inf)
        bounds = (x0_bounds, x1_bounds)
        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解结果为成功，期望的最优值为-80/7，期望的最优解为[-8/7, 18/7]
        _assert_success(res, desired_fun=-80 / 7, desired_x=[-8 / 7, 18 / 7])

    def test_bounds_equal_but_infeasible(self):
        # 定义一个目标函数系数向量
        c = [-4, 1]
        # 创建一个3x2的不等式约束系数矩阵
        A_ub = [[7, -2], [0, 1], [2, -2]]
        # 创建一个长度为3的不等式约束右侧向量
        b_ub = [14, 0, 3]
        # 定义两个变量的上下界，第一个变量上下界相同为2
        bounds = [(2, 2), (0, None)]
        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解结果为不可行
        _assert_infeasible(res)

    def test_bounds_equal_but_infeasible2(self):
        # 定义一个目标函数系数向量
        c = [-4, 1]
        # 创建一个3x2的等式约束系数矩阵
        A_eq = [[7, -2], [0, 1], [2, -2]]
        # 创建一个长度为3的等式约束右侧向量
        b_eq = [14, 0, 3]
        # 定义两个变量的上下界，第一个变量上下界相同为2
        bounds = [(2, 2), (0, None)]
        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解结果为不可行
        _assert_infeasible(res)

    def test_bounds_equal_no_presolve(self):
        # 描述一个Bug情况：当一个变量的上下界相等但未启用presolve时，
        # 变量应被消除，但实际上约束未被消除，导致后处理出现问题。
        # 定义一个目标函数系数向量
        c = [1, 2]
        # 创建一个2x2的不等式约束系数矩阵
        A_ub = [[1, 2], [1.1, 2.2]]
        # 创建一个长度为2的不等式约束右侧向量
        b_ub = [4, 8]
        # 定义两个变量的上下界，第一个变量上下界相同为2
        bounds = [(1, 2), (2, 2)]

        # 复制self.options中的内容到o字典中，并设置presolve为False
        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False

        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        # 断言求解结果为不可行
        _assert_infeasible(res)

    def test_zero_column_1(self):
        # 定义m和n的值分别为3和4
        m, n = 3, 4
        # 设定随机数种子为0
        np.random.seed(0)
        # 创建一个长度为n的随机数向量作为目标函数系数向量
        c = np.random.rand(n)
        # 将第二个变量的目标函数系数设置为1
        c[1] = 1
        # 创建一个mxn的随机数矩阵作为等式约束系数矩阵
        A_eq = np.random.rand(m, n)
        # 将矩阵A_eq的第二列设置为0，即第二个变量不参与等式约束
        A_eq[:, 1] = 0
        # 创建一个长度为m的随机数向量作为等式约束右侧向量
        b_eq = np.random.rand(m)
        # 创建一个1x4的不等式约束系数矩阵
        A_ub = [[1, 0, 1, 1]]
        # 创建一个标量作为不等式约束右侧向量
        b_ub = 3
        # 定义四个变量的上下界
        bounds = [(-10, 10), (-10, 10), (-10, None), (None, None)]
        # 调用线性规划函数求解问题，并传入相应的参数
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 断言求解结果为成功，期望的最优值为-9.7087836730413404
        _assert_success(res, desired_fun=-9.7087836730413404)
    def test_zero_column_2(self):
        if self.method in {'highs-ds', 'highs-ipm'}:
            # 如果使用的方法是 'highs-ds' 或者 'highs-ipm'
            # 查看上游问题 https://github.com/ERGO-Code/HiGHS/issues/648
            pytest.xfail()  # 标记此测试为预期失败

        np.random.seed(0)
        m, n = 2, 4
        c = np.random.rand(n)  # 创建一个大小为 n 的随机向量 c
        c[1] = -1  # 将 c 的第二个元素设置为 -1
        A_eq = np.random.rand(m, n)  # 创建一个大小为 m x n 的随机矩阵 A_eq
        A_eq[:, 1] = 0  # 将 A_eq 的第二列所有元素设置为 0
        b_eq = np.random.rand(m)  # 创建一个大小为 m 的随机向量 b_eq

        A_ub = np.random.rand(m, n)  # 创建一个大小为 m x n 的随机矩阵 A_ub
        A_ub[:, 1] = 0  # 将 A_ub 的第二列所有元素设置为 0
        b_ub = np.random.rand(m)  # 创建一个大小为 m 的随机向量 b_ub
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)  # 断言问题无界

        # 在 presolve 中检测到无界问题
        if self.options.get('presolve', True) and "highs" not in self.method:
            # 如果启用了 presolve 并且方法不包含 'highs'
            # HiGHS 在 presolve 中检测到无界或不可行问题
            # 需要进行一次单纯形迭代以确保问题无界
            # 其他求解器会在可行时报告问题无界
            assert_equal(res.nit, 0)

    def test_zero_row_1(self):
        c = [1, 2, 3]  # 创建一个简单的目标函数向量 c
        A_eq = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]  # 创建一个包含零行的约束矩阵 A_eq
        b_eq = [0, 3, 0]  # 创建一个约束向量 b_eq
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3)  # 断言问题成功，期望的目标函数值为 3

    def test_zero_row_2(self):
        A_ub = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]  # 创建一个包含零行的不等式约束矩阵 A_ub
        b_ub = [0, 3, 0]  # 创建一个不等式约束向量 b_ub
        c = [1, 2, 3]  # 创建一个目标函数向量 c
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0)  # 断言问题成功，期望的目标函数值为 0

    def test_zero_row_3(self):
        m, n = 2, 4
        c = np.random.rand(n)  # 创建一个大小为 n 的随机向量 c
        A_eq = np.random.rand(m, n)  # 创建一个大小为 m x n 的随机矩阵 A_eq
        A_eq[0, :] = 0  # 将 A_eq 的第一行所有元素设置为 0
        b_eq = np.random.rand(m)  # 创建一个大小为 m 的随机向量 b_eq
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)  # 断言问题不可行

        # 在 presolve 中检测到不可行问题
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_zero_row_4(self):
        m, n = 2, 4
        c = np.random.rand(n)  # 创建一个大小为 n 的随机向量 c
        A_ub = np.random.rand(m, n)  # 创建一个大小为 m x n 的随机矩阵 A_ub
        A_ub[0, :] = 0  # 将 A_ub 的第一行所有元素设置为 0
        b_ub = -np.random.rand(m)  # 创建一个大小为 m 的随机向量 b_ub
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)  # 断言问题不可行

        # 在 presolve 中检测到不可行问题
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_eq_1(self):
        c = [1, 1, 1, 2]  # 创建一个目标函数向量 c
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]  # 创建一个包含单个约束行的矩阵 A_eq
        b_eq = [1, 2, 2, 4]  # 创建一个约束向量 b_eq
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)  # 断言问题不可行

        # 在 presolve 中检测到不可行问题
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)
    def test_singleton_row_eq_2(self):
        c = [1, 1, 1, 2]  # 目标函数的系数向量
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]  # 等式约束的系数矩阵
        b_eq = [1, 2, 1, 4]  # 等式约束的右侧常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_success(res, desired_fun=4)  # 断言求解成功，并验证目标函数值是否为期望值

    def test_singleton_row_ub_1(self):
        c = [1, 1, 1, 2]  # 目标函数的系数向量
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]  # 不等式约束的系数矩阵
        b_ub = [1, 2, -2, 4]  # 不等式约束的右侧常数向量
        bounds = [(None, None), (0, None), (0, None), (0, None)]  # 变量的上下界
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_infeasible(res)  # 断言问题无法解决

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)  # 如果启用了预处理，断言迭代次数为0

    def test_singleton_row_ub_2(self):
        c = [1, 1, 1, 2]  # 目标函数的系数向量
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]  # 不等式约束的系数矩阵
        b_ub = [1, 2, -0.5, 4]  # 不等式约束的右侧常数向量
        bounds = [(None, None), (0, None), (0, None), (0, None)]  # 变量的上下界
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_success(res, desired_fun=0.5)  # 断言求解成功，并验证目标函数值是否为期望值

    def test_infeasible(self):
        # Test linprog response to an infeasible problem
        c = [-1, -1]  # 目标函数的系数向量
        A_ub = [[1, 0], [0, 1], [-1, -1]]  # 不等式约束的系数矩阵
        b_ub = [2, 2, -5]  # 不等式约束的右侧常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_infeasible(res)  # 断言问题无法解决

    def test_infeasible_inequality_bounds(self):
        c = [1]  # 目标函数的系数向量
        A_ub = [[2]]  # 不等式约束的系数矩阵
        b_ub = 4  # 不等式约束的右侧常数向量
        bounds = (5, 6)  # 变量的上下界
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_infeasible(res)  # 断言问题无法解决

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)  # 如果启用了预处理，断言迭代次数为0

    def test_unbounded(self):
        # Test linprog response to an unbounded problem
        c = np.array([1, 1]) * -1  # 最大化目标函数，系数向量取负号
        A_ub = [[-1, 1], [-1, -1]]  # 不等式约束的系数矩阵
        b_ub = [-1, -2]  # 不等式约束的右侧常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解问题
        _assert_unbounded(res)  # 断言问题为无界的

    def test_unbounded_below_no_presolve_corrected(self):
        c = [1]  # 目标函数的系数向量
        bounds = [(None, 1)]  # 变量的上下界

        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False  # 禁用预处理选项

        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options=o)  # 调用线性规划求解器求解问题
        if self.method == "revised simplex":
            # Revised simplex has a special pathway for no constraints.
            assert_equal(res.status, 5)  # 如果使用修订单纯形法，断言状态为5（无约束问题）
        else:
            _assert_unbounded(res)  # 断言问题为无界的
    def test_unbounded_no_nontrivial_constraints_1(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        # 定义优化目标函数的系数向量 c
        c = np.array([0, 0, 0, 1, -1, -1])
        # 定义不等式约束矩阵 A_ub
        A_ub = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, -1]])
        # 定义不等式约束右侧的常数向量 b_ub
        b_ub = np.array([2, -2, 0])
        # 定义变量的边界范围 bounds
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (0, None)]
        # 调用线性规划求解器进行优化，返回结果对象 res
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 调用断言函数，验证结果 res 是否无界
        _assert_unbounded(res)
        # 如果方法不以 "highs" 开头，进行额外的断言
        if not self.method.lower().startswith("highs"):
            # 断言最后一个变量的最优值为正无穷
            assert_equal(res.x[-1], np.inf)
            # 断言消息的前36个字符表明问题是（显然）无界的
            assert_equal(res.message[:36],
                         "The problem is (trivially) unbounded")

    def test_unbounded_no_nontrivial_constraints_2(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        # 定义优化目标函数的系数向量 c
        c = np.array([0, 0, 0, 1, -1, 1])
        # 定义不等式约束矩阵 A_ub
        A_ub = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]])
        # 定义不等式约束右侧的常数向量 b_ub
        b_ub = np.array([2, -2, 0])
        # 定义变量的边界范围 bounds
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (None, 0)]
        # 调用线性规划求解器进行优化，返回结果对象 res
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 调用断言函数，验证结果 res 是否无界
        _assert_unbounded(res)
        # 如果方法不以 "highs" 开头，进行额外的断言
        if not self.method.lower().startswith("highs"):
            # 断言最后一个变量的最优值为负无穷
            assert_equal(res.x[-1], -np.inf)
            # 断言消息的前36个字符表明问题是（显然）无界的
            assert_equal(res.message[:36],
                         "The problem is (trivially) unbounded")

    def test_cyclic_recovery(self):
        # 测试线性规划在 Klee-Minty 问题中的循环恢复能力
        # Klee-Minty 问题详情见 https://www.math.ubc.ca/~israel/m340/kleemin3.pdf
        # 定义优化目标函数的系数向量 c，进行最大化
        c = np.array([100, 10, 1]) * -1
        # 定义不等式约束矩阵 A_ub
        A_ub = [[1, 0, 0],
                [20, 1, 0],
                [200, 20, 1]]
        # 定义不等式约束右侧的常数向量 b_ub
        b_ub = [1, 100, 10000]
        # 调用线性规划求解器进行优化，返回结果对象 res
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 调用断言函数，验证结果 res 是否成功，期望最优解为 [0, 0, 10000]
        _assert_success(res, desired_x=[0, 0, 10000], atol=5e-6, rtol=1e-7)
    def test_cyclic_bland(self):
        # 测试 Bland 规则在循环问题上的效果
        c = np.array([-10, 57, 9, 24.])
        A_ub = np.array([[0.5, -5.5, -2.5, 9],
                         [0.5, -1.5, -0.5, 1],
                         [1, 0, 0, 0]])
        b_ub = [0, 0, 1]

        # 复制现有的选项字典但修改 maxiter
        maxiter = 100
        o = {key: val for key, val in self.options.items()}
        o['maxiter'] = maxiter

        # 使用指定的线性规划方法和选项解决问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)

        if self.method == 'simplex' and not self.options.get('bland'):
            # 对于使用单纯形法且未启用 Bland 规则的情况，检查是否达到迭代上限
            _assert_iteration_limit_reached(res, o['maxiter'])
        else:
            # 对于其他方法，包括启用 Bland 规则的单纯形法，验证是否成功解决
            _assert_success(res, desired_x=[1, 0, 1, 0])
        # 需要注意，修正单纯形法会跳过此测试，因为它可能会或可能不会出现循环，这依赖于初始基础

    def test_remove_redundancy_infeasibility(self):
        # 主要测试冗余移除，这在 test__remove_redundancy.py 中已经仔细测试过
        m, n = 10, 10
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        b_eq = np.random.rand(m)
        A_eq[-1, :] = 2 * A_eq[-2, :]
        b_eq[-1] *= -1
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            # 解决带有给定线性规划方法和选项的问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        # 验证结果为不可行
        _assert_infeasible(res)

    #################
    # General Tests #
    #################

    def test_nontrivial_problem(self):
        # 问题涉及所有类型的约束，负资源限制和舍入问题
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 使用指定的线性规划方法和选项解决问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 验证结果成功，期望的目标函数值为 f_star，期望的解为 x_star
        _assert_success(res, desired_fun=f_star, desired_x=x_star)

    def test_lpgen_problem(self):
        # 使用较大的问题测试 linprog（400个变量，40个约束），由 https://gist.github.com/denis-bz/8647461 生成
        A_ub, b_ub, c = lpgen_2d(20, 20)

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            # 使用指定的线性规划方法和选项解决问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        # 验证结果成功，期望的目标函数值为 -64.049494229
        _assert_success(res, desired_fun=-64.049494229)
    def test_network_flow(self):
        # A network flow problem with supply and demand at nodes
        # and with costs along directed edges.
        # https://www.princeton.edu/~rvdb/542/lectures/lec10.pdf
        
        # 定义成本数组 c，表示每条边的成本
        c = [2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18]
        
        # 定义节点数和极值定义
        n, p = -1, 1
        
        # 等式约束矩阵 A_eq，表示节点流量平衡约束
        A_eq = [
            [n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0],
            [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0],
            [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0],
            [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]]
        
        # 等式约束向量 b_eq，表示节点流量平衡约束值
        b_eq = [0, 19, -16, 33, 0, 0, -36]
        
        # 使用 suppress_warnings 上下文管理器来屏蔽特定警告
        with suppress_warnings() as sup:
            sup.filter(LinAlgWarning)  # 屏蔽线性代数警告
            # 使用线性规划函数 linprog 求解网络流问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        
        # 断言结果是否成功，并设定期望的目标函数值和容差
        _assert_success(res, desired_fun=755, atol=1e-6, rtol=1e-7)

    def test_network_flow_limited_capacity(self):
        # A network flow problem with supply and demand at nodes
        # and with costs and capacities along directed edges.
        # http://blog.sommer-forst.de/2013/04/10/
        
        # 定义成本数组 c，表示每条边的成本
        c = [2, 2, 1, 3, 1]
        
        # 边界条件 bounds，表示每条边的容量上下界
        bounds = [
            [0, 4],
            [0, 2],
            [0, 2],
            [0, 3],
            [0, 5]]
        
        # 定义节点数和极值定义
        n, p = -1, 1
        
        # 等式约束矩阵 A_eq，表示节点流量平衡约束
        A_eq = [
            [n, n, 0, 0, 0],
            [p, 0, n, n, 0],
            [0, p, p, 0, n],
            [0, 0, 0, p, p]]
        
        # 等式约束向量 b_eq，表示节点流量平衡约束值
        b_eq = [-4, 0, 0, 4]

        # 使用 suppress_warnings 上下文管理器来屏蔽特定警告
        with suppress_warnings() as sup:
            # 如果有 UmfpackWarning，则屏蔽之
            if has_umfpack:
                sup.filter(UmfpackWarning)
            # 屏蔽特定的运行时警告和优化警告
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Solving system with option...")
            sup.filter(LinAlgWarning)
            
            # 使用线性规划函数 linprog 求解网络流问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        
        # 断言结果是否成功，并设定期望的目标函数值
        _assert_success(res, desired_fun=14)

    def test_simplex_algorithm_wikipedia_example(self):
        # https://en.wikipedia.org/wiki/Simplex_algorithm#Example
        
        # 定义成本数组 c，表示每个变量的成本系数
        c = [-2, -3, -4]
        
        # 不等式约束矩阵 A_ub，表示每个不等式约束的系数
        A_ub = [
            [3, 2, 1],
            [2, 5, 3]]
        
        # 不等式约束向量 b_ub，表示每个不等式约束的右侧值
        b_ub = [10, 15]
        
        # 使用线性规划函数 linprog 求解问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        
        # 断言结果是否成功，并设定期望的目标函数值
        _assert_success(res, desired_fun=-20)
    def test_enzo_example(self):
        # https://github.com/scipy/scipy/issues/1779 lp2.py
        #
        # 从以下Octave代码翻译而来:
        # http://www.ecs.shimane-u.ac.jp/~kyoshida/lpeng.htm
        # 并在原作者 Prof. Kazunobu Yoshida 的明确许可下，
        # 由 Enzo Michelangeli 以 MIT 许可发布
        c = [4, 8, 3, 0, 0, 0]  # 定义优化问题的目标函数系数
        A_eq = [  # 定义等式约束的系数矩阵
            [2, 5, 3, -1, 0, 0],
            [3, 2.5, 8, 0, -1, 0],
            [8, 10, 4, 0, 0, -1]]
        b_eq = [185, 155, 600]  # 等式约束右侧的常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解
        _assert_success(res, desired_fun=317.5,
                        desired_x=[66.25, 0, 17.5, 0, 183.75, 0],  # 断言优化结果的期望目标函数值和解向量
                        atol=6e-6, rtol=1e-7)

    def test_enzo_example_b(self):
        # 从 https://github.com/scipy/scipy/pull/218 救援而来
        c = [2.8, 6.3, 10.8, -2.8, -6.3, -10.8]  # 定义优化问题的目标函数系数
        A_eq = [[-1, -1, -1, 0, 0, 0],  # 定义等式约束的系数矩阵
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1]]
        b_eq = [-0.5, 0.4, 0.3, 0.3, 0.3]  # 等式约束右侧的常数向量

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)  # 调用线性规划求解器求解
        _assert_success(res, desired_fun=-1.77,
                        desired_x=[0.3, 0.2, 0.0, 0.0, 0.1, 0.3])  # 断言优化结果的期望目标函数值和解向量

    def test_enzo_example_c_with_degeneracy(self):
        # 从 https://github.com/scipy/scipy/pull/218 救援而来
        m = 20
        c = -np.ones(m)  # 定义优化问题的目标函数系数
        tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))  # 定义等式约束的系数矩阵
        b_eq = [0, 0]  # 等式约束右侧的常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解
        _assert_success(res, desired_fun=0, desired_x=np.zeros(m))  # 断言优化结果的期望目标函数值和解向量

    def test_enzo_example_c_with_unboundedness(self):
        # 从 https://github.com/scipy/scipy/pull/218 救援而来
        m = 50
        c = -np.ones(m)  # 定义优化问题的目标函数系数
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        # 此测试依赖于 `cos(0) -1 == sin(0)`，因此确保这是正确的
        # (SIMD 代码或 -ffast-math 可能导致虚假失败)
        row0 = np.cos(tmp) - 1
        row0[0] = 0.0
        row1 = np.sin(tmp)
        row1[0] = 0.0
        A_eq = np.vstack((row0, row1))  # 定义等式约束的系数矩阵
        b_eq = [0, 0]  # 等式约束右侧的常数向量
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)  # 调用线性规划求解器求解
        _assert_unbounded(res)  # 断言优化问题为无界
    def test_enzo_example_c_with_infeasibility(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        定义测试函数，用于测试特定情况下的线性规划问题，从指定的 GitHub PR 中恢复该示例
        m = 50
        初始化向量 c，包含 m 个 -1 元素
        c = -np.ones(m)
        根据 m 值生成向量 tmp，用于构建等式约束 A_eq
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        构建 A_eq 矩阵，其中包含两行，表示余弦和正弦函数
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        定义等式约束的右侧向量 b_eq
        b_eq = [1, 1]

        从 self.options 中复制所有选项到字典 o
        o = {key: self.options[key] for key in self.options}
        将选项中的 "presolve" 设为 False
        o["presolve"] = False

        调用线性规划函数 linprog 进行求解
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        调用断言函数 _assert_infeasible 对结果进行验证

    def test_basic_artificial_vars(self):
        # Problem is chosen to test two phase simplex methods when at the end
        # of phase 1 some artificial variables remain in the basis.
        # Also, for `method='simplex'`, the row in the tableau corresponding
        # with the artificial variables is not all zero.
        定义测试函数，用于测试基本的人工变量问题，特别是在第一阶段结束时基础中仍存在一些人工变量时的两阶段单纯形法
        初始化目标函数系数向量 c
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
        初始化不等式约束的系数矩阵 A_ub
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0],
                         [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
                         [1.0, 1.0, 0, 0, 0, 0]])
        初始化不等式约束的右侧向量 b_ub
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        初始化等式约束的系数矩阵 A_eq
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        初始化等式约束的右侧向量 b_eq
        b_eq = np.array([0, 0])
        调用线性规划函数 linprog 进行求解
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        调用断言函数 _assert_success 对结果进行验证，期望目标函数值为 0，解向量为全零向量

    def test_optimize_result(self):
        # check all fields in OptimizeResult
        定义测试函数，用于检查 OptimizeResult 中的所有字段
        从 very_random_gen(0) 生成随机的线性规划问题数据 c, A_ub, b_ub, A_eq, b_eq, bounds
        调用线性规划函数 linprog 进行求解
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        使用断言 assert_ 验证结果的成功标志 res.success
        使用断言 assert_ 验证迭代次数 res.nit
        使用断言 assert_ 验证状态标志 res.status
        如果当前方法不是 'highs'，则使用断言 assert_ 验证消息 res.message 的内容
        使用断言 assert_allclose 验证目标函数值的计算结果
        使用断言 assert_allclose 验证等式约束的残差
        使用断言 assert_allclose 验证不等式约束的残差
        遍历字段名为 'eqlin', 'ineqlin', 'lower', 'upper' 的结果对象 res.keys()
            如果结果中包含该字段，则使用断言 assert 验证其边际值和残差的类型为 np.ndarray

    #################
    # Bug Fix Tests #
    #################
    def test_bug_6139(self):
        # 定义一个测试方法，用于验证 linprog(method='simplex') 在某些情况下无法找到基本可行解的问题
        # 如果第一阶段的伪目标函数超出给定的容差(tol)，则会失败
        # 参考：https://github.com/scipy/scipy/issues/6139

        # 注意：这不严格属于 bug，因为默认的容差确定了结果是否足够接近零，并且不应期望对所有情况都有效。

        c = np.array([1, 1, 1])  # 定义优化目标函数的系数向量
        A_eq = np.array([[1., 0., 0.], [-1000., 0., - 1000.]])  # 定义等式约束的系数矩阵
        b_eq = np.array([5.00000000e+00, -1.00000000e+04])  # 定义等式约束的右侧向量
        A_ub = -np.array([[0., 1000000., 1010000.]])  # 定义不等式约束的系数矩阵
        b_ub = -np.array([10000000.])  # 定义不等式约束的右侧向量
        bounds = (None, None)  # 定义变量的上下界限制

        # 调用 linprog 函数进行线性规划求解
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        # 断言求解成功，并验证期望的目标函数值和解向量
        _assert_success(res, desired_fun=14.95,
                        desired_x=np.array([5, 4.95, 5]))

    def test_bug_6690(self):
        # 定义一个测试方法，用于验证 linprog simplex 方法在报告成功的情况下依然可能违反边界约束的问题
        # 参考：https://github.com/scipy/scipy/issues/6690

        A_eq = np.array([[0, 0, 0, 0.93, 0, 0.65, 0, 0, 0.83, 0]])  # 定义等式约束的系数矩阵
        b_eq = np.array([0.9626])  # 定义等式约束的右侧向量
        A_ub = np.array([
            [0, 0, 0, 1.18, 0, 0, 0, -0.2, 0, -0.22],  # 定义不等式约束的系数矩阵的第一行
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 定义不等式约束的系数矩阵的第二行
            [0, 0, 0, 0.43, 0, 0, 0, 0, 0, 0],  # 定义不等式约束的系数矩阵的第三行
            [0, -1.22, -0.25, 0, 0, 0, -2.06, 0, 0, 1.37],  # 定义不等式约束的系数矩阵的第四行
            [0, 0, 0, 0, 0, 0, 0, -0.25, 0, 0]  # 定义不等式约束的系数矩阵的第五行
        ])
        b_ub = np.array([0.615, 0, 0.172, -0.869, -0.022])  # 定义不等式约束的右侧向量
        bounds = np.array([
            [-0.84, -0.97, 0.34, 0.4, -0.33, -0.74, 0.47, 0.09, -1.45, -0.73],  # 定义变量的下界
            [0.37, 0.02, 2.86, 0.86, 1.18, 0.5, 1.76, 0.17, 0.32, -0.15]  # 定义变量的上界
        ]).T
        c = np.array([
            -1.64, 0.7, 1.8, -1.06, -1.16, 0.26, 2.13, 1.53, 0.66, 0.28  # 定义优化目标函数的系数向量
            ])

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)  # 屏蔽 UMFpack 的警告信息
            sup.filter(OptimizeWarning,
                       "Solving system with option 'cholesky'")  # 屏蔽优化警告信息：使用选项 'cholesky' 求解系统
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")  # 屏蔽优化警告信息：使用选项 'sym_pos' 求解系统
            sup.filter(RuntimeWarning, "invalid value encountered")  # 屏蔽运行时警告信息：遇到无效值
            sup.filter(LinAlgWarning)  # 屏蔽线性代数警告信息
            # 调用 linprog 函数进行线性规划求解
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)

        desired_fun = -1.19099999999  # 期望的优化目标函数值
        desired_x = np.array([0.3700, -0.9700, 0.3400, 0.4000, 1.1800,
                              0.5000, 0.4700, 0.0900, 0.3200, -0.7300])  # 期望的解向量
        # 断言求解成功，并验证期望的目标函数值和解向量
        _assert_success(res, desired_fun=desired_fun, desired_x=desired_x)

        # 添加一个小的容差值，确保数组中的值小于或等于上下界
        atol = 1e-6
        assert_array_less(bounds[:, 0] - atol, res.x)  # 断言 res.x 中的值大于等于 bounds 的下界
        assert_array_less(res.x, bounds[:, 1] + atol)  # 断言 res.x 中的值小于等于 bounds 的上界
    def test_bug_7237(self):
        # 此函数测试 GitHub 上的 issue 7237
        # 当线性规划的单纯形法在主元值非常接近零时可能会出现“爆炸”的情况
        # https://github.com/scipy/scipy/issues/7237

        c = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0])
        # 系数矩阵 A_ub
        A_ub = np.array([
            [1., -724., 911., -551., -555., -896., 478., -80., -293.],
            [1., 566., 42., 937., 233., 883., 392., -909., 57.],
            [1., -208., -894., 539., 321., 532., -924., 942., 55.],
            [1., 857., -859., 83., 462., -265., -971., 826., 482.],
            [1., 314., -424., 245., -424., 194., -443., -104., -429.],
            [1., 540., 679., 361., 149., -827., 876., 633., 302.],
            [0., -1., -0., -0., -0., -0., -0., -0., -0.],
            [0., -0., -1., -0., -0., -0., -0., -0., -0.],
            [0., -0., -0., -1., -0., -0., -0., -0., -0.],
            [0., -0., -0., -0., -1., -0., -0., -0., -0.],
            [0., -0., -0., -0., -0., -1., -0., -0., -0.],
            [0., -0., -0., -0., -0., -0., -1., -0., -0.],
            [0., -0., -0., -0., -0., -0., -0., -1., -0.],
            [0., -0., -0., -0., -0., -0., -0., -0., -1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ])
        # 不等式约束的右侧向量 b_ub
        b_ub = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.])
        # 等式约束的系数矩阵 A_eq
        A_eq = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1.]])
        # 等式约束的右侧向量 b_eq
        b_eq = np.array([[1.]])
        # 决策变量的上下界 bounds
        bounds = [(None, None)] * 9

        # 调用线性规划函数 linprog 进行计算
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 验证最优解的函数值是否符合期望值
        _assert_success(res, desired_fun=108.568535, atol=1e-6)
    def test_bug_8174(self):
        # 测试问题编号为 8174 的bug
        # https://github.com/scipy/scipy/issues/8174
        # 当枢纽值接近零时，单纯形法有时会“爆炸”。

        # 定义不等式约束矩阵 A_ub
        A_ub = np.array([
            [22714, 1008, 13380, -2713.5, -1116],
            [-4986, -1092, -31220, 17386.5, 684],
            [-4986, 0, 0, -2713.5, 0],
            [22714, 0, 0, 17386.5, 0]])

        # 定义不等式约束向量 b_ub
        b_ub = np.zeros(A_ub.shape[0])

        # 定义目标函数系数向量 c
        c = -np.ones(A_ub.shape[1])

        # 定义变量的取值范围 bounds
        bounds = [(0, 1)] * A_ub.shape[1]

        # 使用 suppress_warnings 上下文管理器，过滤特定的运行时警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            # 调用线性规划求解器 linprog，返回结果对象 res
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)

        # 如果选项中的 tol 值小于 1e-10，并且使用的方法是 'simplex' 单纯形法
        if self.options.get('tol', 1e-9) < 1e-10 and self.method == 'simplex':
            # 断言无法找到基本可行解
            _assert_unable_to_find_basic_feasible_sol(res)
        else:
            # 断言求解成功，期望的目标函数值为 -2.0080717488789235，允许误差为 1e-6
            _assert_success(res, desired_fun=-2.0080717488789235, atol=1e-6)

    def test_bug_8174_2(self):
        # 测试问题编号为 8174 的附加示例
        # https://github.com/scipy/scipy/issues/8174
        # https://stackoverflow.com/questions/47717012/linprog-in-scipy-optimize-checking-solution

        # 定义目标函数系数向量 c
        c = np.array([1, 0, 0, 0, 0, 0, 0])

        # 定义不等式约束矩阵 A_ub
        A_ub = -np.identity(7)

        # 定义不等式约束向量 b_ub
        b_ub = np.array([[-2], [-2], [-2], [-2], [-2], [-2], [-2]])

        # 定义等式约束矩阵 A_eq
        A_eq = np.array([
            [1, 1, 1, 1, 1, 1, 0],
            [0.3, 1.3, 0.9, 0, 0, 0, -1],
            [0.3, 0, 0, 0, 0, 0, -2/3],
            [0, 0.65, 0, 0, 0, 0, -1/15],
            [0, 0, 0.3, 0, 0, 0, -1/15]
        ])

        # 定义等式约束向量 b_eq
        b_eq = np.array([[100], [0], [0], [0], [0]])

        # 使用 suppress_warnings 上下文管理器，过滤特定的运行时警告
        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            # 调用线性规划求解器 linprog，返回结果对象 res
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        
        # 断言求解成功，期望的目标函数值为 43.3333333331385
        _assert_success(res, desired_fun=43.3333333331385)

    def test_bug_8561(self):
        # 测试当使用 Bland 规则时选择正确的枢纽行
        # 这原本是为了单纯形法与 Bland 规则编写的，但测试所有方法/选项也无妨
        # https://github.com/scipy/scipy/issues/8561

        # 定义目标函数系数向量 c
        c = np.array([7, 0, -4, 1.5, 1.5])

        # 定义不等式约束矩阵 A_ub
        A_ub = np.array([
            [4, 5.5, 1.5, 1.0, -3.5],
            [1, -2.5, -2, 2.5, 0.5],
            [3, -0.5, 4, -12.5, -7],
            [-1, 4.5, 2, -3.5, -2],
            [5.5, 2, -4.5, -1, 9.5]])

        # 定义不等式约束向量 b_ub
        b_ub = np.array([0, 0, 0, 0, 1])

        # 调用线性规划求解器 linprog，返回结果对象 res
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=self.options,
                      method=self.method)

        # 断言求解成功，期望的解向量为 [0, 0, 19, 16/3, 29/3]
        _assert_success(res, desired_x=[0, 0, 19, 16/3, 29/3])
    def test_bug_8662(self):
        # 测试 GitHub 上的 Issue 8662，检查 linprog simplex 是否报告不正确的最优结果
        c = [-10, 10, 6, 3]  # 目标函数的系数向量
        A_ub = [[8, -8, -4, 6],  # 不等式约束的系数矩阵
                [-8, 8, 4, -6],
                [-4, 4, 8, -4],
                [3, -3, -3, -10]]
        b_ub = [9, -9, -9, -4]  # 不等式约束的右侧常数
        bounds = [(0, None), (0, None), (0, None), (0, None)]  # 变量的上下界
        desired_fun = 36.0000000000  # 期望的最优目标函数值

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)  # 抑制特定的警告类
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res1 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method, options=self.options)
        
        # 将边界条件作为一个约束添加
        A_ub.append([0, 0, -1, 0])  # 添加新的约束系数
        b_ub.append(0)  # 添加新的约束右侧常数
        bounds[2] = (None, None)  # 修改第三个变量的边界

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)  # 抑制特定的警告类
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res2 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method, options=self.options)
        
        rtol = 1e-5  # 相对误差的阈值
        _assert_success(res1, desired_fun=desired_fun, rtol=rtol)  # 断言第一次优化的成功性
        _assert_success(res2, desired_fun=desired_fun, rtol=rtol)  # 断言第二次优化的成功性

    def test_bug_8663(self):
        # 暴露了 presolve 中的一个 bug
        # https://github.com/scipy/scipy/issues/8663
        c = [1, 5]  # 目标函数的系数向量
        A_eq = [[0, -7]]  # 等式约束的系数矩阵
        b_eq = [-6]  # 等式约束的右侧常数
        bounds = [(0, None), (None, None)]  # 变量的上下界
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 6./7], desired_fun=5*6./7)  # 断言优化结果的成功性及期望解和目标函数值

    def test_bug_8664(self):
        # 当 presolve 关闭时，interior-point 在处理这个问题时出现了问题
        # 在 TestLinprogIPSpecific 中测试 interior-point 时关闭 presolve 的情况
        # https://github.com/scipy/scipy/issues/8664
        c = [4]  # 目标函数的系数向量
        A_ub = [[2], [5]]  # 不等式约束的系数矩阵
        b_ub = [4, 4]  # 不等式约束的右侧常数
        A_eq = [[0], [-8], [9]]  # 等式约束的系数矩阵
        b_eq = [3, 2, 10]  # 等式约束的右侧常数
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)  # 抑制特定的警告类
            sup.filter(OptimizeWarning, "Solving system with option...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_infeasible(res)  # 断言问题是不可行的
    def test_bug_8973(self):
        """
        Test whether bug described at:
        https://github.com/scipy/scipy/issues/8973
        was fixed.
        """
        # 定义一个线性规划问题的系数向量 c
        c = np.array([0, 0, 0, 1, -1])
        # 定义不等式约束的系数矩阵 A_ub
        A_ub = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        # 定义不等式约束的右侧向量 b_ub
        b_ub = np.array([2, -2])
        # 定义变量的上下界 bounds
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1)]
        # 调用线性规划求解器 linprog 解决问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 使用自定义函数检查求解结果是否成功，并期望目标函数值为 -2
        _assert_success(res, desired_fun=-2)
        # 检查是否满足 c @ res.x == res.fun，即最优解对应的目标函数值
        assert_equal(c @ res.x, res.fun)

    def test_bug_8973_2(self):
        """
        Additional test for:
        https://github.com/scipy/scipy/issues/8973
        suggested in
        https://github.com/scipy/scipy/pull/8985
        review by @antonior92
        """
        # 定义一个简化的线性规划问题
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        # 调用线性规划求解器 linprog 解决问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # 使用自定义函数检查求解结果是否成功，并期望最优解为 [-2]，目标函数值为 0
        _assert_success(res, desired_x=[-2], desired_fun=0)

    def test_bug_10124(self):
        """
        Test for linprog docstring problem
        'disp'=True caused revised simplex failure
        """
        # 定义一个简化的线性规划问题
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        # 重新定义 c, A_ub, b_ub, bounds，以更复杂的形式
        c = [-1, 4]
        A_ub = [[-3, 1], [1, 2]]
        b_ub = [6, 4]
        bounds = [(None, None), (-3, None)]
        # 定义优化选项 o，将 self.options 合并进去
        o = {"disp": True}
        o.update(self.options)
        # 调用线性规划求解器 linprog 解决问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        # 使用自定义函数检查求解结果是否成功，并期望最优解为 [10, -3]，目标函数值为 -22
        _assert_success(res, desired_x=[10, -3], desired_fun=-22)

    def test_bug_10349(self):
        """
        Test for redundancy removal tolerance issue
        https://github.com/scipy/scipy/issues/10349
        """
        # 定义等式约束的系数矩阵 A_eq 和右侧向量 b_eq
        A_eq = np.array([[1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1]])
        b_eq = np.array([221, 210, 10, 141, 198, 102])
        # 定义目标函数的系数向量 c
        c = np.concatenate((0, 1, np.zeros(4)), axis=None)
        # 使用 suppress_warnings 上下文管理器，过滤特定警告类型
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            # 调用线性规划求解器 linprog 解决问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        # 使用自定义函数检查求解结果是否成功，并期望最优解为 [129, 92, 12, 198, 0, 10]，目标函数值为 92
        _assert_success(res, desired_x=[129, 92, 12, 198, 0, 10], desired_fun=92)

    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason=("Failing on some local macOS builds, "
                                "see gh-13846"))
    def test_bug_10466(self):
        """
        Test that autoscale fixes poorly-scaled problem
        """
        # 定义一个包含问题数据的系数向量 c
        c = [-8., -0., -8., -0., -8., -0., -0., -0., -0., -0., -0., -0., -0.]
        # 定义包含等式约束的系数矩阵 A_eq
        A_eq = [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
        # 定义包含等式约束的右侧常数向量 b_eq
        b_eq = [3.14572800e+08, 4.19430400e+08, 5.24288000e+08,
                1.00663296e+09, 1.07374182e+09, 1.07374182e+09,
                1.07374182e+09, 1.07374182e+09, 1.07374182e+09,
                1.07374182e+09]

        # 初始化空字典 o
        o = {}
        # 如果当前方法不是以 "highs" 开头，则设置 autoscale 选项为 True
        if not self.method.startswith("highs"):
            o = {"autoscale": True}
        # 更新字典 o，将 self.options 的内容合并进去
        o.update(self.options)

        # 使用 suppress_warnings 上下文管理器
        with suppress_warnings() as sup:
            # 过滤掉特定的警告类型
            sup.filter(OptimizeWarning, "Solving system with option...")
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(RuntimeWarning, "divide by zero encountered...")
            sup.filter(RuntimeWarning, "overflow encountered...")
            sup.filter(RuntimeWarning, "invalid value encountered...")
            sup.filter(LinAlgWarning, "Ill-conditioned matrix...")
            # 调用 linprog 函数解决线性规划问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=o)
        # 断言求解结果的目标函数值与预期值的接近程度
        assert_allclose(res.fun, -8589934560)

    def test_bug_20584(self):
        """
        Test that when integrality is a list of all zeros, linprog gives the
        same result as when it is an array of all zeros / integrality=None
        """
        # 定义目标函数系数向量 c
        c = [1, 1]
        # 定义不等式约束的系数矩阵 A_ub 和右侧常数向量 b_ub
        A_ub = [[-1, 0]]
        b_ub = [-2.5]
        # 使用不同的 integrality 参数调用 linprog 函数并获取结果
        res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, integrality=[0, 0])
        res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, integrality=np.asarray([0, 0]))
        res3 = linprog(c, A_ub=A_ub, b_ub=b_ub, integrality=None)
        # 断言不同参数下求解结果的变量值 x 相等
        assert_equal(res1.x, res2.x)
        assert_equal(res1.x, res3.x)
#########################
# Method-specific Tests #
#########################


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class LinprogSimplexTests(LinprogCommonTests):
    method = "simplex"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class LinprogIPTests(LinprogCommonTests):
    method = "interior-point"

    def test_bug_10466(self):
        # 跳过这个测试，因为解算器已经被弃用且测试失败
        pytest.skip("Test is failing, but solver is deprecated.")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class LinprogRSTests(LinprogCommonTests):
    method = "revised simplex"

    # Revised simplex does not reliably solve these problems.
    # Failure is intermittent due to the random choice of elements to complete
    # the basis after phase 1 terminates. In any case, linprog exists
    # gracefully, reporting numerical difficulties. I do not think this should
    # prevent revised simplex from being merged, as it solves the problems
    # most of the time and solves a broader range of problems than the existing
    # simplex implementation.
    # I believe that the root cause is the same for all three and that this
    # same issue prevents revised simplex from solving many other problems
    # reliably. Somehow the pivoting rule allows the algorithm to pivot into
    # a singular basis. I haven't been able to find a reference that
    # acknowledges this possibility, suggesting that there is a bug. On the
    # other hand, the pivoting rule is quite simple, and I can't find a
    # mistake, which suggests that this is a possibility with the pivoting
    # rule. Hopefully, a better pivoting rule will fix the issue.

    def test_bug_5400(self):
        # 跳过这个测试，因为出现间歇性失败是可以接受的
        pytest.skip("Intermittent failure acceptable.")

    def test_bug_8662(self):
        # 跳过这个测试，因为出现间歇性失败是可以接受的
        pytest.skip("Intermittent failure acceptable.")

    def test_network_flow(self):
        # 跳过这个测试，因为出现间歇性失败是可以接受的
        pytest.skip("Intermittent failure acceptable.")


class LinprogHiGHSTests(LinprogCommonTests):
    def test_callback(self):
        # this is the problem from test_callback
        # 定义一个回调函数cb，用于测试回调功能
        def cb(res):
            return None
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        # 断言调用linprog时会抛出NotImplementedError异常，验证未实现的特性
        assert_raises(NotImplementedError, linprog, c, A_ub=A_ub, b_ub=b_ub,
                      callback=cb, method=self.method)
        # 调用linprog求解优化问题，并验证结果的正确性
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=self.method)
        _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])

    @pytest.mark.parametrize("options",
                             [{"maxiter": -1},
                              {"disp": -1},
                              {"presolve": -1},
                              {"time_limit": -1},
                              {"dual_feasibility_tolerance": -1},
                              {"primal_feasibility_tolerance": -1},
                              {"ipm_optimality_tolerance": -1},
                              {"simplex_dual_edge_weight_strategy": "ekki"},
                              ])
    # 定义一个测试方法，用于测试无效选项值
    def test_invalid_option_values(self, options):
        # 定义内部函数 f，用于调用 linprog 函数，传入参数 method 和 options
        def f(options):
            linprog(1, method=self.method, options=options)
        # 更新 options 字典
        options.update(self.options)
        # 断言调用 f 函数会触发 OptimizeWarning 异常，传入 options 参数
        assert_warns(OptimizeWarning, f, options=options)

    # 定义测试交叉验证的方法
    def test_crossover(self):
        # 调用 magic_square 函数获取 A_eq, b_eq, c 等参数
        A_eq, b_eq, c, _, _ = magic_square(4)
        # 设定变量 bounds 区间为 (0, 1)
        bounds = (0, 1)
        # 调用 linprog 函数进行线性规划，传入参数 c, A_eq, b_eq, bounds, method 和 options
        res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        # 断言如果 method 不是 "highs-ipm"，则 crossover_nit 应为非零
        assert_equal(res.crossover_nit == 0, self.method != "highs-ipm")

    # 使用 pytest.mark.fail_slow(10) 标记的测试方法
    def test_marginals(self):
        # 通过 very_random_gen 函数生成 c, A_ub, b_ub, A_eq, b_eq, bounds 参数
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=0)
        # 调用 linprog 函数进行线性规划，传入参数 c, A_ub, b_ub, A_eq, b_eq, bounds, method 和 options
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        # 将 bounds 的上下界分别赋给 lb 和 ub
        lb, ub = bounds.T

        # 对 b_ub 的灵敏度
        def f_bub(x):
            return linprog(c, A_ub, x, A_eq, b_eq, bounds,
                           method=self.method).fun

        # 使用 '3-point' 方法对 f_bub 函数求 b_ub 的导数近似值 dfdbub
        dfdbub = approx_derivative(f_bub, b_ub, method='3-point', f0=res.fun)
        # 断言 res.ineqlin.marginals 与 dfdbub 的近似值相等
        assert_allclose(res.ineqlin.marginals, dfdbub)

        # 对 b_eq 的灵敏度
        def f_beq(x):
            return linprog(c, A_ub, b_ub, A_eq, x, bounds,
                           method=self.method).fun

        # 使用 '3-point' 方法对 f_beq 函数求 b_eq 的导数近似值 dfdbeq
        dfdbeq = approx_derivative(f_beq, b_eq, method='3-point', f0=res.fun)
        # 断言 res.eqlin.marginals 与 dfdbeq 的近似值相等
        assert_allclose(res.eqlin.marginals, dfdbeq)

        # 对 lb 的灵敏度
        def f_lb(x):
            # 将 lb 与 ub 组合成 bounds 数组
            bounds = np.array([x, ub]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method).fun

        # 使用 '3-point' 方法对 f_lb 函数求 lb 的导数近似值 dfdlb
        with np.errstate(invalid='ignore'):
            dfdlb = approx_derivative(f_lb, lb, method='3-point', f0=res.fun)
            dfdlb[~np.isfinite(lb)] = 0

        # 断言 res.lower.marginals 与 dfdlb 的近似值相等
        assert_allclose(res.lower.marginals, dfdlb)

        # 对 ub 的灵敏度
        def f_ub(x):
            # 将 lb 与 ub 组合成 bounds 数组
            bounds = np.array([lb, x]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method).fun

        # 使用 '3-point' 方法对 f_ub 函数求 ub 的导数近似值 dfdub
        with np.errstate(invalid='ignore'):
            dfdub = approx_derivative(f_ub, ub, method='3-point', f0=res.fun)
            dfdub[~np.isfinite(ub)] = 0

        # 断言 res.upper.marginals 与 dfdub 的近似值相等
        assert_allclose(res.upper.marginals, dfdub)
    # 定义一个测试函数，用于验证解是否满足对偶可行性
    def test_dual_feasibility(self):
        # 使用指定种子生成非常随机的线性规划问题数据
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        # 求解线性规划问题，返回结果对象 res
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)

        # 根据文献中的 KKT 对偶可行性方程，计算残差
        resid = (-c + A_ub.T @ res.ineqlin.marginals +
                 A_eq.T @ res.eqlin.marginals +
                 res.upper.marginals +
                 res.lower.marginals)
        # 断言残差应该接近于 0，允许的误差为 1e-12
        assert_allclose(resid, 0, atol=1e-12)

    # 定义一个测试函数，用于验证互补松弛条件是否满足
    def test_complementary_slackness(self):
        # 使用指定种子生成非常随机的线性规划问题数据
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        # 求解线性规划问题，返回结果对象 res
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)

        # 根据文献中的 KKT 互补松弛条件方程，对于非零的右侧向量进行修改
        assert np.allclose(res.ineqlin.marginals @ (b_ub - A_ub @ res.x), 0)
################################
# Simplex Option-Specific Tests#
################################


class TestLinprogSimplexDefault(LinprogSimplexTests):

    def setup_method(self):
        self.options = {}  # 初始化空字典，用于存储测试选项

    def test_bug_5400(self):
        pytest.skip("Simplex fails on this problem.")  # 跳过测试并添加说明信息

    def test_bug_7237_low_tol(self):
        # 如果容差设置过低，则测试失败。在这里，测试即使解决方案错误，也会引发适当的错误。
        pytest.skip("Simplex fails on this problem.")

    def test_bug_8174_low_tol(self):
        # 如果容差设置过低，则测试失败。在这里，测试即使解决方案错误，也会引发适当的警告。
        self.options.update({'tol': 1e-12})  # 更新测试选项中的容差值
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()  # 调用父类方法并期望警告信息


class TestLinprogSimplexBland(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'bland': True}  # 初始化选项字典，启用 Bland 规则

    def test_bug_5400(self):
        pytest.skip("Simplex fails on this problem.")  # 跳过测试并添加说明信息

    def test_bug_8174_low_tol(self):
        # 如果容差设置过低，则测试失败。在这里，测试即使解决方案错误，也会引发适当的错误。
        self.options.update({'tol': 1e-12})  # 更新测试选项中的容差值
        with pytest.raises(AssertionError):
            with pytest.warns(OptimizeWarning):
                super().test_bug_8174()  # 调用父类方法并期望错误和警告信息


class TestLinprogSimplexNoPresolve(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'presolve': False}  # 初始化选项字典，禁用预处理步骤

    is_32_bit = np.intp(0).itemsize < 8  # 检测是否为32位系统
    is_linux = sys.platform.startswith('linux')  # 检测是否为Linux系统

    @pytest.mark.xfail(
        condition=is_32_bit and is_linux,
        reason='Fails with warning on 32-bit linux')
    def test_bug_5400(self):
        super().test_bug_5400()  # 调用父类方法，期望该测试在特定条件下失败并给出原因

    def test_bug_6139_low_tol(self):
        # Linprog(method='simplex') fails to find a basic feasible solution
        # if phase 1 pseudo-objective function is outside the provided tol.
        # https://github.com/scipy/scipy/issues/6139
        # Without ``presolve`` eliminating such rows the result is incorrect.
        self.options.update({'tol': 1e-12})  # 更新测试选项中的容差值
        with pytest.raises(AssertionError, match='linprog status 4'):
            return super().test_bug_6139()  # 调用父类方法，期望引发特定错误匹配的异常信息

    def test_bug_7237_low_tol(self):
        pytest.skip("Simplex fails on this problem.")  # 跳过测试并添加说明信息

    def test_bug_8174_low_tol(self):
        # 如果容差设置过低，则测试失败。在这里，测试即使解决方案错误，也会引发适当的警告。
        self.options.update({'tol': 1e-12})  # 更新测试选项中的容差值
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()  # 调用父类方法并期望警告信息

    def test_unbounded_no_nontrivial_constraints_1(self):
        pytest.skip("Tests behavior specific to presolve")  # 跳过测试并添加说明信息

    def test_unbounded_no_nontrivial_constraints_2(self):
        pytest.skip("Tests behavior specific to presolve")  # 跳过测试并添加说明信息


#######################################
# Interior-Point Option-Specific Tests#
#######################################

# 定义一个测试类 TestLinprogIPDense，继承自 LinprogIPTests
class TestLinprogIPDense(LinprogIPTests):
    # 设置选项字典，表示稀疏性为 False
    options = {"sparse": False}

    # 根据平台进行条件标记，跳过测试
    @pytest.mark.skipif(
        sys.platform == 'darwin',
        reason="Fails on some macOS builds for reason not relevant to test"
    )
    # 定义测试方法 test_bug_6139，调用父类方法执行相同测试
    def test_bug_6139(self):
        super().test_bug_6139()

# 如果有 cholmod 模块可用，则定义测试类 TestLinprogIPSparseCholmod
if has_cholmod:
    class TestLinprogIPSparseCholmod(LinprogIPTests):
        # 设置选项字典，表示稀疏性为 True，使用 Cholmod 进行 Cholesky 分解
        options = {"sparse": True, "cholesky": True}

# 如果有 umfpack 模块可用，则定义测试类 TestLinprogIPSparseUmfpack
if has_umfpack:
    class TestLinprogIPSparseUmfpack(LinprogIPTests):
        # 设置选项字典，表示稀疏性为 True，不使用 Cholesky 分解
        options = {"sparse": True, "cholesky": False}

        # 定义测试方法 test_network_flow_limited_capacity，跳过测试并给出原因
        def test_network_flow_limited_capacity(self):
            pytest.skip("Failing due to numerical issues on some platforms.")

# 定义测试类 TestLinprogIPSparse，继承自 LinprogIPTests
class TestLinprogIPSparse(LinprogIPTests):
    # 设置选项字典，表示稀疏性为 True，不使用 Cholesky 分解，不要求对称正定
    options = {"sparse": True, "cholesky": False, "sym_pos": False}

    # 根据平台进行条件标记，跳过测试
    @pytest.mark.skipif(
        sys.platform == 'darwin',
        reason="Fails on macOS x86 Accelerate builds (gh-20510)"
    )
    # 标记预期测试失败，并给出失败原因
    @pytest.mark.xfail_on_32bit("This test is sensitive to machine epsilon level "
                                "perturbations in linear system solution in "
                                "_linprog_ip._sym_solve.")
    # 定义测试方法 test_bug_6139，调用父类方法执行相同测试
    def test_bug_6139(self):
        super().test_bug_6139()

    # 标记预期测试失败，并给出失败原因
    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    # 定义测试方法 test_bug_6690，调用父类方法执行相同测试
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super().test_bug_6690()

    # 定义测试方法 test_magic_square_sparse_no_presolve，测试带有秩缺陷 A_eq 矩阵的问题
    def test_magic_square_sparse_no_presolve(self):
        # 调用 magic_square 函数获取问题的约束矩阵和向量
        A_eq, b_eq, c, _, _ = magic_square(3)
        # 设置变量的取值范围
        bounds = (0, 1)

        # 使用 suppress_warnings 上下文管理器
        with suppress_warnings() as sup:
            # 如果有 umfpack 模块可用，则过滤 UmfpackWarning
            if has_umfpack:
                sup.filter(UmfpackWarning)
            # 过滤 MatrixRankWarning，提醒“矩阵完全奇异”
            sup.filter(MatrixRankWarning, "Matrix is exactly singular")
            # 过滤 OptimizeWarning，提示“正在使用选项解决系统…”
            sup.filter(OptimizeWarning, "Solving system with option...")

            # 复制选项字典中的值，添加不使用预处理的选项
            o = {key: self.options[key] for key in self.options}
            o["presolve"] = False

            # 调用 linprog 函数求解线性规划问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=o)
        # 调用 _assert_success 函数，断言求解成功，期望函数值为 1.730550597
        _assert_success(res, desired_fun=1.730550597)
    def test_sparse_solve_options(self):
        # 定义测试函数，用于检查使用所有列置换选项解决问题
        A_eq, b_eq, c, _, _ = magic_square(3)
        # 使用抑制警告上下文管理器，忽略特定优化警告
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Invalid permc_spec option")
            # 复制测试对象的选项字典
            o = {key: self.options[key] for key in self.options}
            # 不同的列置换规范
            permc_specs = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A',
                           'COLAMD', 'ekki-ekki-ekki')
            # 对于每个列置换规范进行迭代
            for permc_spec in permc_specs:
                # 设置当前列置换规范到选项字典
                o["permc_spec"] = permc_spec
                # 调用线性规划函数解决问题，并获取结果
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                              method=self.method, options=o)
                # 断言解决成功，期望函数值为 1.730550597
                _assert_success(res, desired_fun=1.730550597)
class TestLinprogIPSparsePresolve(LinprogIPTests):
    options = {"sparse": True, "_sparse_presolve": True}  # 设置测试选项，启用稀疏求解和稀疏预处理

    @pytest.mark.skipif(
        sys.platform == 'darwin',
        reason="Fails on macOS x86 Accelerate builds (gh-20510)"
    )
    @pytest.mark.xfail_on_32bit("This test is sensitive to machine epsilon level "
                                "perturbations in linear system solution in "
                                "_linprog_ip._sym_solve.")
    def test_bug_6139(self):
        super().test_bug_6139()  # 调用父类的测试方法来执行 bug 6139 的测试

    def test_enzo_example_c_with_infeasibility(self):
        pytest.skip('_sparse_presolve=True incompatible with presolve=False')  # 跳过测试，因为 sparse_presolve=True 与 presolve=False 不兼容

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super().test_bug_6690()  # 调用父类的测试方法来执行 bug 6690 的测试


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestLinprogIPSpecific:
    method = "interior-point"  # 设定测试方法为 interior-point

    # the following tests don't need to be performed separately for
    # sparse presolve, sparse after presolve, and dense

    def test_solver_select(self):
        # check that default solver is selected as expected
        if has_cholmod:
            options = {'sparse': True, 'cholesky': True}  # 如果有 cholmod 库，则使用稀疏和 cholesky 分解
        elif has_umfpack:
            options = {'sparse': True, 'cholesky': False}  # 如果有 umfpack 库，则使用稀疏但不使用 cholesky 分解
        else:
            options = {'sparse': True, 'cholesky': False, 'sym_pos': False}  # 否则，稀疏但不使用 cholesky 分解和对称正定性
        A, b, c = lpgen_2d(20, 20)  # 生成测试用的线性规划问题数据
        res1 = linprog(c, A_ub=A, b_ub=b, method=self.method, options=options)  # 调用线性规划求解器，使用指定的选项
        res2 = linprog(c, A_ub=A, b_ub=b, method=self.method)  # 调用线性规划求解器，使用默认的选项
        assert_allclose(res1.fun, res2.fun,
                        err_msg="linprog default solver unexpected result",
                        rtol=2e-15, atol=1e-15)  # 检查两种方式求解的结果是否接近

    def test_unbounded_below_no_presolve_original(self):
        # formerly caused segfault in TravisCI w/ "cholesky":True
        c = [-1]  # 设置目标函数系数
        bounds = [(None, 1)]  # 设置变量的边界条件
        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options={"presolve": False, "cholesky": True})  # 调用线性规划求解器，禁用预处理并使用 cholesky 分解
        _assert_success(res, desired_fun=-1)  # 检查求解结果是否成功，并验证目标函数值是否正确

    def test_cholesky(self):
        # use cholesky factorization and triangular solves
        A, b, c = lpgen_2d(20, 20)  # 生成测试用的线性规划问题数据
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"cholesky": True})  # 调用线性规划求解器，使用 cholesky 分解
        _assert_success(res, desired_fun=-64.049494229)  # 检查求解结果是否成功，并验证目标函数值是否正确
    # 定义一个测试方法，用于测试改进的初始点
    def test_alternate_initial_point(self):
        # 使用"改进"的初始点生成二维线性规划问题的系数和约束
        A, b, c = lpgen_2d(20, 20)
        
        # 使用上下文管理器抑制特定的警告消息
        with suppress_warnings() as sup:
            # 过滤特定类型的运行时警告
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            # 过滤优化过程中的警告消息
            sup.filter(OptimizeWarning, "Solving system with option...")
            # 过滤线性代数相关的警告消息
            sup.filter(LinAlgWarning, "Ill-conditioned matrix...")
            
            # 调用线性规划求解器求解问题
            res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                          options={"ip": True, "disp": True})
            
            # IP（内点）代码与稠密/稀疏无关
            # 以上注释说明内点方法的求解过程独立于稠密或稀疏矩阵的存储方式

        # 断言求解成功，并验证期望的目标函数值
        _assert_success(res, desired_fun=-64.049494229)

    # 定义另一个测试方法，用于测试 Bug 8664
    def test_bug_8664(self):
        # 在禁用预处理时，内点方法可能会遇到问题
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        
        # 使用上下文管理器抑制特定的警告消息
        with suppress_warnings() as sup:
            # 过滤运行时警告
            sup.filter(RuntimeWarning)
            # 过滤优化过程中的警告消息
            sup.filter(OptimizeWarning, "Solving system with option...")
            
            # 调用线性规划求解器求解问题
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options={"presolve": False})
        
        # 断言求解未成功，即期望报告未成功求解
        assert_(not res.success, "Incorrectly reported success")
########################################
# Revised Simplex Option-Specific Tests#
########################################

# 创建一个测试类 TestLinprogRSCommon，继承自 LinprogRSTests
class TestLinprogRSCommon(LinprogRSTests):
    # 定义空的选项字典
    options = {}

    # 跳过测试，注明这个测试偶尔失败是可以接受的
    def test_cyclic_bland(self):
        pytest.skip("Intermittent failure acceptable.")

    # 测试一个非平凡的线性规划问题，使用预设的初始猜测解
    def test_nontrivial_problem_with_guess(self):
        # 调用 nontrivial_problem 函数获取问题的参数和预期结果
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 调用 linprog 函数求解线性规划问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        # 断言函数执行成功，并验证函数值和解的正确性
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        # 断言迭代次数为0
        assert_equal(res.nit, 0)

    # 测试一个非平凡的线性规划问题，使用未界定变量的初始猜测解
    def test_nontrivial_problem_with_unbounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 设置变量的界限
        bounds = [(None, None), (None, None), (0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    # 测试一个非平凡的线性规划问题，使用有界变量的初始猜测解
    def test_nontrivial_problem_with_bounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 设置变量的界限
        bounds = [(None, 1), (1, None), (0, None), (.4, .6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    # 测试一个非平凡的线性规划问题，包含一个负无界变量的初始猜测解
    def test_nontrivial_problem_with_negative_unbounded_variable(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 更改等式约束的右侧值和预期解
        b_eq = [4]
        x_star = np.array([-219/385, 582/385, 0, 4/10])
        f_star = 3951/385
        # 设置变量的界限
        bounds = [(None, None), (1, None), (0, None), (.4, .6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    # 测试一个非平凡的线性规划问题，使用一个错误的初始猜测解
    def test_nontrivial_problem_with_bad_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 使用错误的初始猜测解
        bad_guess = [1, 2, 3, .5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=bad_guess)
        # 断言函数状态为6，表示错误的初始猜测
        assert_equal(res.status, 6)
    # 定义一个测试方法，用于测试带有猜测解的冗余约束情况
    def test_redundant_constraints_with_guess(self):
        # 调用 magic_square 函数生成一个大小为3的魔方阵的系数矩阵 A、约束向量 b、目标函数系数向量 c
        A, b, c, _, _ = magic_square(3)
        # 生成一个与 c 同样形状的随机数组 p
        p = np.random.rand(*c.shape)
        # 使用 suppress_warnings 上下文管理器，屏蔽一些特定的警告信息
        with suppress_warnings() as sup:
            # 过滤掉 OptimizeWarning 类型的警告信息，内容为"A_eq does not appear..."
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            # 过滤掉 RuntimeWarning 类型的警告信息，内容为"invalid value encountered"
            sup.filter(RuntimeWarning, "invalid value encountered")
            # 过滤掉 LinAlgWarning 类型的警告信息
            sup.filter(LinAlgWarning)
            # 调用 linprog 函数，求解线性规划问题，约束条件为 A_eq=A, b_eq=b，优化方法为 self.method
            res = linprog(c, A_eq=A, b_eq=b, method=self.method)
            # 第二次调用 linprog 函数，利用第一次的结果 res.x 作为初始解，继续求解
            res2 = linprog(c, A_eq=A, b_eq=b, method=self.method, x0=res.x)
            # 第三次调用 linprog 函数，目标函数系数为 c + p，利用第一次的结果 res.x 作为初始解，继续求解
            res3 = linprog(c + p, A_eq=A, b_eq=b, method=self.method, x0=res.x)
        # 对 res2 的结果进行断言，确保其达到期望的目标函数值 1.730550597
        _assert_success(res2, desired_fun=1.730550597)
        # 断言 res2 的迭代次数为 0
        assert_equal(res2.nit, 0)
        # 对 res3 的结果进行断言，确保求解成功
        _assert_success(res3)
        # 断言 res3 的迭代次数少于 res 的迭代次数，即热启动减少了迭代次数
        assert_(res3.nit < res.nit)  # hot start reduces iterations
class TestLinprogRSBland(LinprogRSTests):
    options = {"pivot": "bland"}


# 定义一个测试类 TestLinprogRSBland，继承自 LinprogRSTests，用于测试具有 Bland 规则的线性规划算法
options = {"pivot": "bland"}  # 设置选项，指定使用 Bland 规则作为枢纽元选择策略


############################################
# HiGHS-Simplex-Dual Option-Specific Tests #
############################################


class TestLinprogHiGHSSimplexDual(LinprogHiGHSTests):
    method = "highs-ds"
    options = {}

    def test_lad_regression(self):
        '''
        The scaled model should be optimal, i.e. not produce unscaled model
        infeasible.  See https://github.com/ERGO-Code/HiGHS/issues/494.
        '''
        # 测试确保 gh-13610 已解决（HiGHS 缩放和非缩放模型状态不匹配问题）
        c, A_ub, b_ub, bnds = l1_regression_prob()  # 获取 L1 回归问题的系数、约束、边界
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds,
                      method=self.method, options=self.options)  # 调用线性规划求解器，使用指定的方法和选项
        assert_equal(res.status, 0)  # 断言求解状态为最优解
        assert_(res.x is not None)  # 断言解向量不为空
        assert_(np.all(res.slack > -1e-6))  # 断言所有松弛变量大于指定阈值
        assert_(np.all(res.x <= [np.inf if ub is None else ub
                                 for lb, ub in bnds]))  # 断言解向量在指定边界内
        assert_(np.all(res.x >= [-np.inf if lb is None else lb - 1e-7
                                 for lb, ub in bnds]))  # 断言解向量在指定边界内


###################################
# HiGHS-IPM Option-Specific Tests #
###################################


class TestLinprogHiGHSIPM(LinprogHiGHSTests):
    method = "highs-ipm"
    options = {}


###################################
# HiGHS-MIP Option-Specific Tests #
###################################


class TestLinprogHiGHSMIP:
    method = "highs"
    options = {}

    @pytest.mark.fail_slow(10)  # 标记为失败慢测试，最多重试10次
    @pytest.mark.xfail(condition=(sys.maxsize < 2 ** 32 and
                       platform.system() == "Linux"),
                       run=False,
                       reason="gh-16347")  # 标记为预期失败，如果条件满足且运行在 Linux 系统上，原因是 gh-16347
    def test_mip1(self):
        # 解决非松弛魔方问题
        # 同时检查值是否都是整数 - 它们不总是这样从 HiGHS 中出来
        n = 4
        A, b, c, numbers, M = magic_square(n)  # 调用函数生成魔方问题的系数、约束、目标函数、魔方解、魔方总和
        bounds = [(0, 1)] * len(c)  # 设置变量的上下界
        integrality = [1] * len(c)  # 设置变量为整数要求

        res = linprog(c=c*0, A_eq=A, b_eq=b, bounds=bounds,
                      method=self.method, integrality=integrality)  # 调用线性规划求解器，使用指定的方法和选项

        s = (numbers.flatten() * res.x).reshape(n**2, n, n)  # 计算魔方解的分布情况
        square = np.sum(s, axis=0)  # 计算每行和每列的和
        np.testing.assert_allclose(square.sum(axis=0), M)  # 断言每列和等于总和 M
        np.testing.assert_allclose(square.sum(axis=1), M)  # 断言每行和等于总和 M
        np.testing.assert_allclose(np.diag(square).sum(), M)  # 断言主对角线和等于总和 M
        np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)  # 断言副对角线和等于总和 M

        np.testing.assert_allclose(res.x, np.round(res.x), atol=1e-12)  # 断言解向量是整数
    def test_mip2(self):
        # solve MIP with inequality constraints and all integer constraints
        # source: slide 5,
        # https://www.cs.upc.edu/~erodri/webpage/cps/theory/lp/milp/slides.pdf
        
        # 定义不等式约束的系数矩阵 A_ub
        A_ub = np.array([[2, -2], [-8, 10]])
        # 定义不等式约束的右侧常数 b_ub
        b_ub = np.array([-1, 13])
        # 定义优化目标函数的系数向量 c，并转换为负数，因为 linprog 求最小值
        c = -np.array([1, 1])

        # 定义变量的上下界 bounds，这里设置所有变量都大于等于 0
        bounds = np.array([(0, np.inf)] * len(c))
        # 定义整数约束，所有变量均为整数
        integrality = np.ones_like(c)

        # 调用线性规划函数 linprog 求解问题
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, integrality=integrality)

        # 使用 numpy.testing 库检查结果向量 res.x 是否接近于 [1, 2]
        np.testing.assert_allclose(res.x, [1, 2])
        # 使用 numpy.testing 库检查最优解是否接近于 -3
        np.testing.assert_allclose(res.fun, -3)

    def test_mip3(self):
        # solve MIP with inequality constraints and all integer constraints
        # source: https://en.wikipedia.org/wiki/Integer_programming#Example
        
        # 定义不等式约束的系数矩阵 A_ub
        A_ub = np.array([[-1, 1], [3, 2], [2, 3]])
        # 定义不等式约束的右侧常数 b_ub
        b_ub = np.array([1, 12, 12])
        # 定义优化目标函数的系数向量 c，并转换为负数，因为 linprog 求最小值
        c = -np.array([0, 1])

        # 定义变量的上下界 bounds，这里设置所有变量都大于等于 0
        bounds = [(0, np.inf)] * len(c)
        # 定义整数约束，所有变量均为整数
        integrality = [1] * len(c)

        # 调用线性规划函数 linprog 求解问题
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, integrality=integrality)

        # 使用 numpy.testing 库检查最优解是否接近于 -2
        np.testing.assert_allclose(res.fun, -2)
        # 检查两个可能的最优解之一
        assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])

    def test_mip4(self):
        # solve MIP with inequality constraints and only one integer constraint
        # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
        
        # 定义不等式约束的系数矩阵 A_ub
        A_ub = np.array([[-1, -2], [-4, -1], [2, 1]])
        # 定义不等式约束的右侧常数 b_ub
        b_ub = np.array([14, -33, 20])
        # 定义优化目标函数的系数向量 c
        c = np.array([8, 1])

        # 定义变量的上下界 bounds，这里设置所有变量都大于等于 0
        bounds = [(0, np.inf)] * len(c)
        # 定义整数约束，第二个变量为整数
        integrality = [0, 1]

        # 调用线性规划函数 linprog 求解问题
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, integrality=integrality)

        # 使用 numpy.testing 库检查最优解是否接近于 [6.5, 7]
        np.testing.assert_allclose(res.x, [6.5, 7])
        # 使用 numpy.testing 库检查最优目标函数值是否接近于 59
        np.testing.assert_allclose(res.fun, 59)

    def test_mip5(self):
        # solve MIP with inequality and equality constraints
        # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
        
        # 定义不等式约束的系数矩阵 A_ub
        A_ub = np.array([[1, 1, 1]])
        # 定义不等式约束的右侧常数 b_ub
        b_ub = np.array([7])
        # 定义等式约束的系数矩阵 A_eq
        A_eq = np.array([[4, 2, 1]])
        # 定义等式约束的右侧常数 b_eq
        b_eq = np.array([12])
        # 定义优化目标函数的系数向量 c，并转换为负数，因为 linprog 求最小值
        c = np.array([-3, -2, -1])

        # 定义变量的上下界 bounds 和整数约束
        bounds = [(0, np.inf), (0, np.inf), (0, 1)]
        integrality = [0, 1, 0]

        # 调用线性规划函数 linprog 求解问题
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method,
                      integrality=integrality)

        # 使用 numpy.testing 库检查最优解是否接近于 [0, 6, 0]
        np.testing.assert_allclose(res.x, [0, 6, 0])
        # 使用 numpy.testing 库检查最优目标函数值是否接近于 -12
        np.testing.assert_allclose(res.fun, -12)

        # 检查额外的字段是否在结果中存在
        assert res.get("mip_node_count", None) is not None
        assert res.get("mip_dual_bound", None) is not None
        assert res.get("mip_gap", None) is not None

    @pytest.mark.slow
    @pytest.mark.timeout(120)  # 设置测试用例的超时时间为120秒，用于 prerelease_deps_coverage_64bit_blas 任务
    def test_mip6(self):
        # 解决一个具有仅等式约束的较大 MIP 问题
        # 参考来源: https://www.mathworks.com/help/optim/ug/intlinprog.html
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],  # 等式约束的系数矩阵 A_eq
                         [39, 16, 22, 28, 26, 30, 23, 24],
                         [18, 14, 29, 27, 30, 38, 26, 26],
                         [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])  # 等式约束的右侧向量 b_eq
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])  # 目标函数的系数向量 c
    
        bounds = [(0, np.inf)]*8  # 变量的上下界约束
        integrality = [1]*8  # 变量的整数约束
    
        # 使用线性规划方法求解
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=self.method, integrality=integrality)
    
        # 断言目标函数值的近似性
        np.testing.assert_allclose(res.fun, 1854)
    
    @pytest.mark.xslow
    def test_mip_rel_gap_passdown(self):
        # 从 test_mip6 中获取的 MIP，使用不同的 mip_rel_gap 值进行求解
        # 解决一个具有仅等式约束的较大 MIP 问题
        # 参考来源: https://www.mathworks.com/help/optim/ug/intlinprog.html
        A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],  # 等式约束的系数矩阵 A_eq
                         [39, 16, 22, 28, 26, 30, 23, 24],
                         [18, 14, 29, 27, 30, 38, 26, 26],
                         [41, 26, 28, 36, 18, 38, 16, 26]])
        b_eq = np.array([7872, 10466, 11322, 12058])  # 等式约束的右侧向量 b_eq
        c = np.array([2, 10, 13, 17, 7, 5, 7, 3])  # 目标函数的系数向量 c
    
        bounds = [(0, np.inf)]*8  # 变量的上下界约束
        integrality = [1]*8  # 变量的整数约束
    
        mip_rel_gaps = [0.5, 0.25, 0.01, 0.001]  # 不同的 mip_rel_gap 值列表
        sol_mip_gaps = []
        for mip_rel_gap in mip_rel_gaps:
            # 使用线性规划方法求解
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method=self.method,
                          integrality=integrality,
                          options={"mip_rel_gap": mip_rel_gap})
            final_mip_gap = res["mip_gap"]
            # 断言确保最终的 mip_gap 小于或等于提供的 mip_rel_gap
            assert final_mip_gap <= mip_rel_gap
            sol_mip_gaps.append(final_mip_gap)
    
        # 确保 mip_rel_gap 参数实际起到作用
        # 检查解的差异逐渐随 mip_rel_gap 参数单调减少
        # np.diff 计算数组的差值，np.flip 反转数组以得到单调递减的解的差异序列
        gap_diffs = np.diff(np.flip(sol_mip_gaps))
        assert np.all(gap_diffs >= 0)  # 断言所有的解的差异都大于等于0
        assert not np.all(gap_diffs == 0)  # 断言不是所有的解的差异都等于0
    # 定义一个测试方法 test_semi_continuous(self)，用于验证问题 #18106 的解决方案是否正确检查了整数性 > 1 的情况：
    # 即使 0 超出界限，也允许值为 0。

    # 创建包含浮点数的 NumPy 数组 c，表示线性规划的目标函数系数
    c = np.array([1., 1., -1, -1])
    
    # 创建包含 NumPy 数组的边界 bounds，定义了每个变量的取值范围
    bounds = np.array([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])
    
    # 创建包含整数的 NumPy 数组 integrality，指定了哪些变量需要是整数
    integrality = np.array([2, 3, 2, 3])

    # 调用线性规划函数 linprog，传入目标函数系数 c、变量边界 bounds、整数约束 integrality 和解决方法 'highs'
    res = linprog(c, bounds=bounds,
                  integrality=integrality, method='highs')

    # 使用 NumPy 的测试函数 np.testing.assert_allclose 验证结果 res 的 x 值接近于 [0, 0, 1.5, 1]
    np.testing.assert_allclose(res.x, [0, 0, 1.5, 1])
    
    # 检查线性规划结果 res 的状态是否为 0，表示问题已经正确求解
    assert res.status == 0
###########################
# Autoscale-Specific Tests#
###########################

# 忽略 DeprecationWarning 警告，并标记为 pytest 测试类
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class AutoscaleTests:
    # 设置 autoscale 选项为 True
    options = {"autoscale": True}

    # 继承 LinprogCommonTests 中的测试方法
    test_bug_6139 = LinprogCommonTests.test_bug_6139
    test_bug_6690 = LinprogCommonTests.test_bug_6690
    test_bug_7237 = LinprogCommonTests.test_bug_7237

# TestAutoscaleIP 类，继承自 AutoscaleTests
class TestAutoscaleIP(AutoscaleTests):
    # 指定方法为 "interior-point"
    method = "interior-point"

    # 定义 test_bug_6139 测试方法
    def test_bug_6139(self):
        # 设置 options 字典中的 'tol' 值为 1e-10
        self.options['tol'] = 1e-10
        # 调用父类的 test_bug_6139 方法，并返回其结果
        return AutoscaleTests.test_bug_6139(self)

# TestAutoscaleSimplex 类，继承自 AutoscaleTests
class TestAutoscaleSimplex(AutoscaleTests):
    # 指定方法为 "simplex"
    method = "simplex"

# TestAutoscaleRS 类，继承自 AutoscaleTests
class TestAutoscaleRS(AutoscaleTests):
    # 指定方法为 "revised simplex"
    method = "revised simplex"

    # 测试非平凡问题，并使用先验猜测值
    def test_nontrivial_problem_with_guess(self):
        # 调用 nontrivial_problem() 函数，获取问题的各种参数和解
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 使用 linprog 函数求解线性规划问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        # 断言解的成功性，验证期望的函数值和解向量
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        # 断言迭代次数为 0
        assert_equal(res.nit, 0)

    # 测试非平凡问题，并使用错误的猜测值
    def test_nontrivial_problem_with_bad_guess(self):
        # 调用 nontrivial_problem() 函数，获取问题的各种参数和解
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        # 定义错误的猜测值
        bad_guess = [1, 2, 3, .5]
        # 使用 linprog 函数求解线性规划问题
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=bad_guess)
        # 断言解的状态为 6（错误的猜测值导致问题无解）
        assert_equal(res.status, 6)

###########################
# Redundancy Removal Tests#
###########################

# 忽略 DeprecationWarning 警告，并标记为 pytest 测试类
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class RRTests:
    # 指定方法为 "interior-point"
    method = "interior-point"
    # 定义 LCT 为 LinprogCommonTests
    LCT = LinprogCommonTests
    # 以下是一些已有冗余的现有测试
    test_RR_infeasibility = LCT.test_remove_redundancy_infeasibility
    test_bug_10349 = LCT.test_bug_10349
    test_bug_7044 = LCT.test_bug_7044
    test_NFLC = LCT.test_network_flow_limited_capacity
    test_enzo_example_b = LCT.test_enzo_example_b

# TestRRSVD 类，继承自 RRTests
class TestRRSVD(RRTests):
    # 设置 options 字典中的 "rr_method" 为 "SVD"
    options = {"rr_method": "SVD"}

# TestRRPivot 类，继承自 RRTests
class TestRRPivot(RRTests):
    # 设置 options 字典中的 "rr_method" 为 "pivot"
    options = {"rr_method": "pivot"}

# TestRRID 类，继承自 RRTests
class TestRRID(RRTests):
    # 设置 options 字典中的 "rr_method" 为 "ID"
    options = {"rr_method": "ID"}
```