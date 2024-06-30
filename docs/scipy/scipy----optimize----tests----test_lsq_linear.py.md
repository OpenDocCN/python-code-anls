# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_lsq_linear.py`

```
# 导入 pytest 库，用于编写和运行测试
import pytest

# 导入 numpy 库，并从中导入所需的模块和函数
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_

# 导入 scipy.sparse 库中的 rand 和 coo_matrix 函数
from scipy.sparse import rand, coo_matrix
# 导入 scipy.sparse.linalg 中的 aslinearoperator 函数
from scipy.sparse.linalg import aslinearoperator
# 导入 scipy.optimize 库中的 lsq_linear 函数
from scipy.optimize import lsq_linear
# 导入 scipy.optimize._minimize 中的 Bounds 类
from scipy.optimize._minimize import Bounds

# 创建一个 3x2 的 numpy 数组 A
A = np.array([
    [0.171, -0.057],
    [-0.049, -0.248],
    [-0.166, 0.054],
])
# 创建一个包含 3 个元素的 numpy 数组 b
b = np.array([0.074, 1.014, -0.383])

# 定义一个 BaseMixin 类
class BaseMixin:
    # 定义 setup_method 方法，在每个测试方法执行前调用
    def setup_method(self):
        # 初始化一个指定种子的随机数生成器
        self.rnd = np.random.RandomState(0)

    # 定义 test_dense_no_bounds 方法，测试密集矩阵且无边界条件下的最小二乘求解
    def test_dense_no_bounds(self):
        # 遍历每个最小二乘求解器 lsq_solver
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 函数求解最小二乘问题
            res = lsq_linear(A, b, method=self.method, lsq_solver=lsq_solver)
            # 断言最小二乘求解结果 res.x 与 lstsq 函数的结果的一致性
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
            # 断言最小二乘求解结果 res.x 与无边界条件时的解的一致性
            assert_allclose(res.x, res.unbounded_sol[0])
    def test_dense_bounds(self):
        # Solutions for comparison are taken from MATLAB.

        # 定义下界 lb 和上界 ub
        lb = np.array([-1, -10])
        ub = np.array([1, 0])

        # 计算无约束解 unbounded_sol
        unbounded_sol = lstsq(A, b, rcond=-1)[0]

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与无约束解的接近程度
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        # 更新下界 lb，其中第一个元素为 0.0，第二个元素为负无穷
        lb = np.array([0.0, -np.inf])

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与预期结果接近，其中第一个元素为 0.0，第二个元素为 -4.084174437334673
            assert_allclose(res.x, np.array([0.0, -4.084174437334673]),
                            atol=1e-6)

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        # 更新下界 lb，其中第一个元素为 -1，第二个元素为 0
        lb = np.array([-1, 0])

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与预期结果接近，其中第一个元素为 0.448427311733504，第二个元素为 0
            assert_allclose(res.x, np.array([0.448427311733504, 0]),
                            atol=1e-15)

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        # 更新上界 ub，其中第一个元素为正无穷，第二个元素为 -5
        ub = np.array([np.inf, -5])

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与预期结果接近，其中第一个元素为 -0.105560998682388，第二个元素为 -5
            assert_allclose(res.x, np.array([-0.105560998682388, -5]))

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        # 更新上界 ub，其中第一个元素为 -1，第二个元素为正无穷
        ub = np.array([-1, np.inf])

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与预期结果接近，其中第一个元素为 -1，第二个元素为 -4.181102129483254
            assert_allclose(res.x, np.array([-1, -4.181102129483254]))

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        # 定义下界 lb 和上界 ub，其中第一个元素为 0，第二个元素为 -4
        lb = np.array([0, -4])
        ub = np.array([1, 0])

        # 对每个最小二乘求解器 lsq_solver 进行测试
        for lsq_solver in self.lsq_solvers:
            # 使用 lsq_linear 求解带约束的最小二乘问题
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)

            # 断言求解结果与预期结果接近，其中第一个元素为 0.005236663400791，第二个元素为 -4
            assert_allclose(res.x, np.array([0.005236663400791, -4]))

            # 断言无约束解的第一个元素与预期 unbounded_sol 接近
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

    def test_bounds_variants(self):
        # 定义向量 x，矩阵 A 和向量 b
        x = np.array([1, 3])
        A = self.rnd.uniform(size=(2, 2))
        b = A @ x

        # 定义下界 lb 和上界 ub，其中每个元素均为 1
        lb = np.array([1, 1])
        ub = np.array([2, 2])

        # 定义旧的边界 bounds_old 和新的边界 bounds_new
        bounds_old = (lb, ub)
        bounds_new = Bounds(lb, ub)

        # 使用 lsq_linear 分别对旧边界和新边界求解
        res_old = lsq_linear(A, b, bounds_old)
        res_new = lsq_linear(A, b, bounds_new)

        # 断言新边界求解结果与无约束解的第一个元素不完全接近
        assert not np.allclose(res_new.x, res_new.unbounded_sol[0])

        # 断言旧边界求解结果与新边界求解结果接近
        assert_allclose(res_old.x, res_new.x)
    # 定义一个测试方法，用于测试使用 NumPy 矩阵的情况
    def test_np_matrix(self):
        # gh-10711: GitHub issue 10711，标记测试用例的来源
        # 使用 np.testing.suppress_warnings 上下文管理器来抑制警告
        with np.testing.suppress_warnings() as sup:
            # 过滤 PendingDeprecationWarning 类型的警告
            sup.filter(PendingDeprecationWarning)
            # 创建一个 NumPy 矩阵 A
            A = np.matrix([[20, -4, 0, 2, 3], [10, -2, 1, 0, -1]])
        # 创建 NumPy 数组 k
        k = np.array([20, 15])
        # 调用 lsq_linear 函数进行最小二乘法线性求解
        lsq_linear(A, k)

    # 定义一个测试方法，用于测试稠密且秩不足的线性方程组情况
    def test_dense_rank_deficient(self):
        # 创建 NumPy 数组 A 和 b
        A = np.array([[-0.307, -0.184]])
        b = np.array([0.773])
        # 设置下界 lb 和上界 ub
        lb = [-0.1, -0.1]
        ub = [0.1, 0.1]
        # 遍历 lsq_solvers 列表中的求解器
        for lsq_solver in self.lsq_solvers:
            # 调用 lsq_linear 函数进行线性最小二乘法求解
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            # 断言结果中的 x 属性与预期值的接近程度
            assert_allclose(res.x, [-0.1, -0.1])
            # 断言结果中的无界解 unbounded_sol 与 lstsq 函数的结果接近
            assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

        # 创建另一个 NumPy 数组 A 和 b
        A = np.array([
            [0.334, 0.668],
            [-0.516, -1.032],
            [0.192, 0.384],
        ])
        b = np.array([-1.436, 0.135, 0.909])
        # 设置新的下界 lb 和上界 ub
        lb = [0, -1]
        ub = [1, -0.5]
        # 再次遍历 lsq_solvers 列表中的求解器
        for lsq_solver in self.lsq_solvers:
            # 调用 lsq_linear 函数进行线性最小二乘法求解
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            # 断言结果中的最优性 optimality 接近于 0
            assert_allclose(res.optimality, 0, atol=1e-11)
            # 断言结果中的无界解 unbounded_sol 与 lstsq 函数的结果接近
            assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

    # 定义一个测试方法，用于测试完整的 lsq_linear 结果
    def test_full_result(self):
        # 创建 NumPy 数组 lb 和 ub
        lb = np.array([0, -4])
        ub = np.array([1, 0])
        # 调用 lsq_linear 函数进行线性最小二乘法求解
        res = lsq_linear(A, b, (lb, ub), method=self.method)

        # 断言结果中的 x 属性与预期值的接近程度
        assert_allclose(res.x, [0.005236663400791, -4])
        # 断言结果中的无界解 unbounded_sol 与 lstsq 函数的结果接近
        assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

        # 计算残差 r
        r = A.dot(res.x) - b
        # 断言结果中的成本 cost
        assert_allclose(res.cost, 0.5 * np.dot(r, r))
        # 断言结果中的函数值 fun
        assert_allclose(res.fun, r)

        # 断言结果中的最优性 optimality 接近于 0
        assert_allclose(res.optimality, 0.0, atol=1e-12)
        # 断言结果中的活跃掩码 active_mask 符合预期
        assert_equal(res.active_mask, [0, -1])
        # 断言迭代次数 nit 小于 15
        assert_(res.nit < 15)
        # 断言结果的状态 status 符合预期
        assert_(res.status == 1 or res.status == 3)
        # 断言结果的消息 message 是一个字符串
        assert_(isinstance(res.message, str))
        # 断言成功标志 success 为真
        assert_(res.success)

    # 这是用于测试问题 #9982 的测试用例
    def test_almost_singular(self):
        # 创建一个接近奇异的 NumPy 数组 A 和 b
        A = np.array(
            [[0.8854232310355122, 0.0365312146937765, 0.0365312146836789],
             [0.3742460132129041, 0.0130523214078376, 0.0130523214077873],
             [0.9680633871281361, 0.0319366128718639, 0.0319366128718388]])

        b = np.array(
            [0.0055029366538097, 0.0026677442422208, 0.0066612514782381])

        # 调用 lsq_linear 函数进行线性最小二乘法求解
        result = lsq_linear(A, b, method=self.method)
        # 断言结果中的成本 cost 小于 1.1e-8
        assert_(result.cost < 1.1e-8)

    # 标记测试用例为较慢的测试，使用 pytest.mark.xslow
    def test_large_rank_deficient(self):
        # 设定随机数种子为0，以便结果可重复
        np.random.seed(0)
        # 随机生成两个整数 n 和 m，并确保 n <= m
        n, m = np.sort(np.random.randint(2, 1000, size=2))
        m *= 2   # 将 m 扩大，使得 m 远远大于 n
        # 随机生成大小为 [m, n] 的矩阵 A，元素取值范围在 -99 到 99 之间
        A = 1.0 * np.random.randint(-99, 99, size=[m, n])
        # 随机生成大小为 [m] 的向量 b，元素取值范围在 -99 到 99 之间
        b = 1.0 * np.random.randint(-99, 99, size=[m])
        # 随机生成大小为 [2, n] 的矩阵 bounds，元素取值范围在 -99 到 99 之间，并确保 bounds[1, :] > bounds[0, :]
        bounds = 1.0 * np.sort(np.random.randint(-99, 99, size=(2, n)), axis=0)
        bounds[1, :] += 1.0  # 确保上限大于下限

        # 通过复制一些列使得 A 矩阵的秩远远不足
        w = np.random.choice(n, n)  # 随机选择一些列并进行复制
        A = A[:, w]

        # 使用 lsq_linear 函数解最小二乘问题，bvls 方法
        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        # 使用 lsq_linear 函数解最小二乘问题，trf 方法
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        # 计算 bvls 方法和 trf 方法的代价函数值
        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        # 断言两种方法的代价函数值非常接近
        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)

    def test_convergence_small_matrix(self):
        # 给定一个小规模的矩阵 A 和向量 b
        A = np.array([[49.0, 41.0, -32.0],
                      [-19.0, -32.0, -8.0],
                      [-13.0, 10.0, 69.0]])
        b = np.array([-41.0, -90.0, 47.0])
        # 给定边界矩阵 bounds
        bounds = np.array([[31.0, -44.0, 26.0],
                           [54.0, -32.0, 28.0]])

        # 使用 lsq_linear 函数解最小二乘问题，bvls 方法
        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        # 使用 lsq_linear 函数解最小二乘问题，trf 方法
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        # 计算 bvls 方法和 trf 方法的代价函数值
        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        # 断言两种方法的代价函数值非常接近
        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)
class SparseMixin:
    # 定义测试稀疏矩阵和线性操作符的方法
    def test_sparse_and_LinearOperator(self):
        m = 5000  # 定义矩阵的行数
        n = 1000  # 定义矩阵的列数
        A = rand(m, n, random_state=0)  # 生成随机的 m x n 稀疏矩阵 A
        b = self.rnd.randn(m)  # 生成长度为 m 的随机向量 b
        res = lsq_linear(A, b)  # 使用最小二乘法求解 A * x = b
        assert_allclose(res.optimality, 0, atol=1e-6)  # 断言优化度接近 0

        A = aslinearoperator(A)  # 转换 A 为线性操作符
        res = lsq_linear(A, b)  # 使用最小二乘法再次求解 A * x = b
        assert_allclose(res.optimality, 0, atol=1e-6)  # 断言优化度接近 0

    @pytest.mark.fail_slow(10)
    # 标记测试为“慢速失败”，如果失败率超过 10%
    def test_sparse_bounds(self):
        m = 5000  # 定义矩阵的行数
        n = 1000  # 定义矩阵的列数
        A = rand(m, n, random_state=0)  # 生成随机的 m x n 稀疏矩阵 A
        b = self.rnd.randn(m)  # 生成长度为 m 的随机向量 b
        lb = self.rnd.randn(n)  # 生成长度为 n 的随机下界向量 lb
        ub = lb + 1  # 上界向量 ub 是 lb 各元素加 1
        res = lsq_linear(A, b, (lb, ub))  # 使用给定边界求解 A * x = b
        assert_allclose(res.optimality, 0.0, atol=1e-6)  # 断言优化度接近 0

        res = lsq_linear(A, b, (lb, ub), lsmr_tol=1e-13,
                         lsmr_maxiter=1500)  # 使用额外参数调用最小二乘法
        assert_allclose(res.optimality, 0.0, atol=1e-6)  # 断言优化度接近 0

        res = lsq_linear(A, b, (lb, ub), lsmr_tol='auto')  # 使用自动设置的 lsmr_tol 参数
        assert_allclose(res.optimality, 0.0, atol=1e-6)  # 断言优化度接近 0

    def test_sparse_ill_conditioned(self):
        # 创建条件数约为 4 百万的稀疏矩阵
        data = np.array([1., 1., 1., 1. + 1e-6, 1.])
        row = np.array([0, 0, 1, 2, 2])
        col = np.array([0, 2, 1, 0, 2])
        A = coo_matrix((data, (row, col)), shape=(3, 3))

        # 获取精确解
        exact_sol = lsq_linear(A.toarray(), b, lsq_solver='exact')

        # 默认 lsmr 参数不应完全收敛解
        default_lsmr_sol = lsq_linear(A, b, lsq_solver='lsmr')
        with pytest.raises(AssertionError, match=""):
            assert_allclose(exact_sol.x, default_lsmr_sol.x)

        # 通过增加最大 lsmr 迭代次数，可以收敛解
        conv_lsmr = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=10)
        assert_allclose(exact_sol.x, conv_lsmr.x)


class TestTRF(BaseMixin, SparseMixin):
    method = 'trf'
    lsq_solvers = ['exact', 'lsmr']


class TestBVLS(BaseMixin):
    method = 'bvls'
    lsq_solvers = ['exact']


class TestErrorChecking:
    def test_option_lsmr_tol(self):
        # lsmr_tol 参数应接受正浮点数、'auto' 字符串或 None
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=1e-2)
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='auto')
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=None)

        # lsmr_tol 参数如果为负浮点数、不是 'auto' 字符串，或者是整数，应该引发错误
        err_message = "`lsmr_tol` must be None, 'auto', or positive float."
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=-0.1)
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='foo')
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=1)
    # 定义测试方法，测试 lsmr_maxiter 参数的不同设置情况

    # 使用 lsq_linear 函数测试 lsmr_maxiter 设置为正整数或 None 的情况
    _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=1)
    _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=None)

    # 使用 lsq_linear 函数测试 lsmr_maxiter 设置为 0 或负数时是否会引发错误
    # 准备错误信息字符串
    err_message = "`lsmr_maxiter` must be None or positive integer."
    # 使用 pytest 模块检查设置 lsmr_maxiter 为 0 是否引发 ValueError 错误，并匹配错误信息
    with pytest.raises(ValueError, match=err_message):
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=0)
    # 使用 pytest 模块检查设置 lsmr_maxiter 为负数是否引发 ValueError 错误，并匹配错误信息
    with pytest.raises(ValueError, match=err_message):
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=-1)
```