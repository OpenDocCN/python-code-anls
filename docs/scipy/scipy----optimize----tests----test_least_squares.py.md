# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_least_squares.py`

```
# 导入 itertools 模块中的 product 函数，用于生成可迭代对象的笛卡尔积
from itertools import product

# 导入 numpy 库，并从中导入 norm 函数和 linalg 子模块
import numpy as np
from numpy.linalg import norm

# 导入 numpy.testing 模块中的几个断言函数，用于测试和验证数值计算的结果
from numpy.testing import (
    assert_,           # 断言函数，用于验证条件是否为真
    assert_allclose,   # 断言函数，用于验证两个数组或数值是否在指定误差范围内相等
    assert_equal,      # 断言函数，用于验证两个对象是否相等
    suppress_warnings  # 函数装饰器，用于在测试中抑制警告
)

# 导入 pytest 库，并从中导入 raises 函数并起别名为 assert_raises
import pytest
from pytest import raises as assert_raises

# 导入 scipy.sparse 库，并从中导入 issparse 和 lil_matrix 函数
from scipy.sparse import issparse, lil_matrix

# 导入 scipy.sparse.linalg 子模块，并从中导入 aslinearoperator 函数
from scipy.sparse.linalg import aslinearoperator

# 导入 scipy.optimize 模块，并从中导入 least_squares 和 Bounds 类
from scipy.optimize import least_squares, Bounds

# 导入 scipy.optimize._lsq.least_squares 模块中的 IMPLEMENTED_LOSSES 变量
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

# 导入 scipy.optimize._lsq.common 模块中的几个变量和函数
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector


# 定义一个简单的二次函数 fun_trivial，带有默认参数 a=0
def fun_trivial(x, a=0):
    return (x - a)**2 + 5.0


# 定义 fun_trivial 函数的雅可比矩阵 jac_trivial，带有默认参数 a=0.0
def jac_trivial(x, a=0.0):
    return 2 * (x - a)


# 定义一个简单的二维向量函数 fun_2d_trivial
def fun_2d_trivial(x):
    return np.array([x[0], x[1]])


# 定义 fun_2d_trivial 函数的雅可比矩阵 jac_2d_trivial，返回一个 2x2 的单位矩阵
def jac_2d_trivial(x):
    return np.identity(2)


# 定义 Rosenbrock 函数 fun_rosenbrock
def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])


# 定义 fun_rosenbrock 函数的雅可比矩阵 jac_rosenbrock
def jac_rosenbrock(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0]
    ])


# 定义维度不匹配的 Rosenbrock 函数 fun_rosenbrock_bad_dim
def jac_rosenbrock_bad_dim(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0],
        [0.0, 0.0]
    ])


# 定义裁剪后的 Rosenbrock 函数 fun_rosenbrock_cropped
def fun_rosenbrock_cropped(x):
    return fun_rosenbrock(x)[0]


# 定义裁剪后的 Rosenbrock 函数的雅可比矩阵 jac_rosenbrock_cropped
def jac_rosenbrock_cropped(x):
    return jac_rosenbrock(x)[0]


# 当输入 x 是一维数组时，返回一个错误维度的函数 fun_wrong_dimensions
def fun_wrong_dimensions(x):
    return np.array([x, x**2, x**3])


# 当输入 x 是一维数组时，返回一个错误维度的函数 jac_wrong_dimensions
# 使用 np.atleast_3d 将 jac_trivial 的结果至少转换为一个三维数组
def jac_wrong_dimensions(x, a=0.0):
    return np.atleast_3d(jac_trivial(x, a=a))


# 定义一个解偏微分方程边值问题的函数 fun_bvp
def fun_bvp(x):
    n = int(np.sqrt(x.shape[0]))  # 计算方程维数
    u = np.zeros((n + 2, n + 2))  # 创建一个 (n+2)x(n+2) 的全零数组 u
    x = x.reshape((n, n))         # 将输入向量 x 重新形状为 n x n 的二维数组
    # 计算边值问题的结果并展平为一维数组返回
    y = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * x + x**3
    return y.ravel()


# 定义一个类 BroydenTridiagonal，用于处理特定的三对角矩阵问题
class BroydenTridiagonal:
    def __init__(self, n=100, mode='sparse'):
        np.random.seed(0)  # 设置随机种子以复现结果

        self.n = n  # 初始化矩阵维数

        self.x0 = -np.ones(n)  # 初始化起始点为全为 -1 的数组
        self.lb = np.linspace(-2, -1.5, n)  # 创建一个从 -2 到 -1.5 的均匀分布的数组作为下界
        self.ub = np.linspace(-0.8, 0.0, n)  # 创建一个从 -0.8 到 0.0 的均匀分布的数组作为上界

        # 给 lb 和 ub 加上小的随机扰动，增加随机性
        self.lb += 0.1 * np.random.randn(n)
        self.ub += 0.1 * np.random.randn(n)

        # 给 x0 加上小的随机扰动，并确保其在 lb 和 ub 的范围内
        self.x0 += 0.1 * np.random.randn(n)
        self.x0 = make_strictly_feasible(self.x0, self.lb, self.ub)

        if mode == 'sparse':
            # 如果模式是 sparse，则创建一个稀疏的 lil_matrix 对象表示三对角矩阵
            self.sparsity = lil_matrix((n, n), dtype=int)
            i = np.arange(n)
            self.sparsity[i, i] = 1
            i = np.arange(1, n)
            self.sparsity[i, i - 1] = 1
            i = np.arange(n - 1)
            self.sparsity[i, i + 1] = 1

            self.jac = self._jac  # 使用 self._jac 函数作为雅可比矩阵的计算方法
        elif mode == 'operator':
            # 如果模式是 operator，则使用 aslinearoperator 函数将 _jac 函数转换为线性操作符
            self.jac = lambda x: aslinearoperator(self._jac(x))
        elif mode == 'dense':
            # 如果模式是 dense，则不使用稀疏表示，直接返回完整的雅可比矩阵
            self.sparsity = None
            self.jac = lambda x: self._jac(x).toarray()
        else:
            assert_(False)  # 如果模式不是 sparse, operator, dense 中的任意一种，则断言失败

    # 定义类方法 fun，计算 BroydenTridiagonal 问题的目标函数
    def fun(self, x):
        f = (3 - x) * x + 1  # 计算目标函数值
        f[1:] -= x[:-1]      # 修正边界条件
        f[:-1] -= 2 * x[1:]  # 修正边界条件
        return f  # 返回目标函数值的数组
    # 定义一个私有方法 `_jac`，用于计算雅可比矩阵
    def _jac(self, x):
        # 创建一个稀疏的 lil_matrix，大小为 self.n x self.n
        J = lil_matrix((self.n, self.n))
        
        # 设置对角线上的元素，i 为 [0, 1, 2, ..., self.n-1]
        i = np.arange(self.n)
        J[i, i] = 3 - 2 * x
        
        # 设置次对角线上的元素，i 为 [1, 2, ..., self.n-1]
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        
        # 设置上次对角线上的元素，i 为 [0, 1, ..., self.n-2]
        i = np.arange(self.n - 1)
        J[i, i + 1] = -2
        
        # 返回计算得到的雅可比矩阵
        return J
class ExponentialFittingProblem:
    """提供指数拟合问题的数据和函数，形式为 y = a + exp(b * x) + noise。"""

    def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1),
                 n_points=11, random_seed=None):
        # 设定随机数种子
        np.random.seed(random_seed)
        # 设置数据点数量和参数数量
        self.m = n_points
        self.n = 2

        # 初始化参数向量
        self.p0 = np.zeros(2)
        # 在指定范围内生成均匀分布的 x 值
        self.x = np.linspace(x_range[0], x_range[1], n_points)

        # 计算 y 值，模拟拟合数据并添加噪声
        self.y = a + np.exp(b * self.x)
        self.y += noise * np.random.randn(self.m)

        # 随机生成异常值并添加到 y 中
        outliers = np.random.randint(0, self.m, n_outliers)
        self.y[outliers] += 50 * noise * np.random.rand(n_outliers)

        # 存储优化参数的初始猜测值
        self.p_opt = np.array([a, b])

    def fun(self, p):
        # 定义拟合函数的残差
        return p[0] + np.exp(p[1] * self.x) - self.y

    def jac(self, p):
        # 计算拟合函数关于参数的雅可比矩阵
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = self.x * np.exp(p[1] * self.x)
        return J


def cubic_soft_l1(z):
    # 计算 cubic soft L1 损失函数的三个分量
    rho = np.empty((3, z.size))

    t = 1 + z
    rho[0] = 3 * (t**(1/3) - 1)
    rho[1] = t ** (-2/3)
    rho[2] = -2/3 * t**(-5/3)

    return rho


LOSSES = list(IMPLEMENTED_LOSSES.keys()) + [cubic_soft_l1]


class BaseMixin:
    def test_basic(self):
        # 测试基本调用序列是否正常工作
        res = least_squares(fun_trivial, 2., method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)
        assert_allclose(res.fun, fun_trivial(res.x))

    def test_args_kwargs(self):
        # 测试 args 和 kwargs 是否正确传递给函数
        a = 3.0
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(
                    UserWarning,
                    "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'"
                )
                res = least_squares(fun_trivial, 2.0, jac, args=(a,),
                                    method=self.method)
                res1 = least_squares(fun_trivial, 2.0, jac, kwargs={'a': a},
                                    method=self.method)

            assert_allclose(res.x, a, rtol=1e-4)
            assert_allclose(res1.x, a, rtol=1e-4)

            assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                          args=(3, 4,), method=self.method)
            assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                          kwargs={'kaboom': 3}, method=self.method)

    def test_jac_options(self):
        # 测试不同的雅可比选项是否正常工作
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(
                    UserWarning,
                    "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'"
                )
                res = least_squares(fun_trivial, 2.0, jac, method=self.method)
            assert_allclose(res.x, 0, atol=1e-4)

        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac='oops',
                      method=self.method)
    # 定义测试函数，用于测试不同的 max_nfev 参数值
    def test_nfev_options(self):
        for max_nfev in [None, 20]:
            # 调用 least_squares 函数进行优化，使用简单的测试函数 fun_trivial
            res = least_squares(fun_trivial, 2.0, max_nfev=max_nfev,
                                method=self.method)
            # 断言优化结果的 x 值接近于 0，允许的绝对误差为 1e-4
            assert_allclose(res.x, 0, atol=1e-4)

    # 定义测试函数，用于测试不同的 x_scale 参数值
    def test_x_scale_options(self):
        for x_scale in [1.0, np.array([0.5]), 'jac']:
            # 调用 least_squares 函数进行优化，使用简单的测试函数 fun_trivial
            res = least_squares(fun_trivial, 2.0, x_scale=x_scale)
            # 断言优化结果的 x 值接近于 0
            assert_allclose(res.x, 0)
        # 对于不合法的 x_scale 参数，断言会引发 ValueError 异常
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale='auto', method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=-1.0, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=None, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=1.0+2.0j, method=self.method)

    # 定义测试函数，用于测试 diff_step 参数
    def test_diff_step(self):
        # 对于 diff_step 参数的不同设置，进行优化并获取结果
        res1 = least_squares(fun_trivial, 2.0, diff_step=1e-1,
                             method=self.method)
        res2 = least_squares(fun_trivial, 2.0, diff_step=-1e-1,
                             method=self.method)
        res3 = least_squares(fun_trivial, 2.0,
                             diff_step=None, method=self.method)
        # 断言不同设置下的优化结果的 x 值接近于 0
        assert_allclose(res1.x, 0, atol=1e-4)
        assert_allclose(res2.x, 0, atol=1e-4)
        assert_allclose(res3.x, 0, atol=1e-4)
        # 断言 res1 和 res2 的优化结果 x 相等
        assert_equal(res1.x, res2.x)
        # 断言 res1 和 res2 的优化过程调用函数的次数 nfev 相等
        assert_equal(res1.nfev, res2.nfev)

    # 定义测试函数，用于测试不正确的 options 参数使用情况
    def test_incorrect_options_usage(self):
        # 断言使用未定义的选项会引发 TypeError 异常
        assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                      method=self.method, options={'no_such_option': 100})
        # 断言 max_nfev 作为 options 参数会引发 TypeError 异常
        assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                      method=self.method, options={'max_nfev': 100})

    # 定义测试函数，用于测试完整的优化结果
    def test_full_result(self):
        # 使用 least_squares 函数进行优化，详细检查各项优化结果的准确性
        res = least_squares(fun_trivial, 2.0, method=self.method)
        # 断言优化结果的 x 值接近于 0
        assert_allclose(res.x, 0, atol=1e-4)
        # 断言优化结果的 cost 接近于 12.5
        assert_allclose(res.cost, 12.5)
        # 断言优化结果的 fun 值接近于 5
        assert_allclose(res.fun, 5)
        # 断言优化结果的 jac 值接近于 0
        assert_allclose(res.jac, 0, atol=1e-4)
        # 断言优化结果的 grad 值接近于 0，允许的绝对误差为 1e-2
        assert_allclose(res.grad, 0, atol=1e-2)
        # 断言优化结果的 optimality 值接近于 0，允许的绝对误差为 1e-2
        assert_allclose(res.optimality, 0, atol=1e-2)
        # 断言优化结果的 active_mask 值为 0
        assert_equal(res.active_mask, 0)
        # 根据不同的优化方法，断言优化过程中函数调用的次数 nfev 和 njev 符合预期
        if self.method == 'lm':
            assert_(res.nfev < 30)
            assert_(res.njev is None)
        else:
            assert_(res.nfev < 10)
            assert_(res.njev < 10)
        # 断言优化结果的状态 status 大于 0，表示成功
        assert_(res.status > 0)
        # 断言优化成功
        assert_(res.success)
    # 测试在最大迭代次数为1时的完整结果。对于 'lm' 方法，直接返回，不进行测试。
    def test_full_result_single_fev(self):
        if self.method == 'lm':
            return

        # 使用最小二乘法计算，期望得到的结果如下：
        # - x 的值应为 [2]
        # - cost 应为 40.5
        # - fun 应为 [9]
        # - jac 应为 [[4]]
        # - grad 应为 [36]
        # - optimality 应为 36
        # - active_mask 应为 [0]
        # - nfev 应为 1
        # - njev 应为 1
        # - status 应为 0
        # - success 应为 0
        res = least_squares(fun_trivial, 2.0, method=self.method,
                            max_nfev=1)
        assert_equal(res.x, np.array([2]))
        assert_equal(res.cost, 40.5)
        assert_equal(res.fun, np.array([9]))
        assert_equal(res.jac, np.array([[4]]))
        assert_equal(res.grad, np.array([36]))
        assert_equal(res.optimality, 36)
        assert_equal(res.active_mask, np.array([0]))
        assert_equal(res.nfev, 1)
        assert_equal(res.njev, 1)
        assert_equal(res.status, 0)
        assert_equal(res.success, 0)

    # 测试 Rosenbrock 函数的优化结果是否接近全局最优点 [1, 1]
    def test_rosenbrock(self):
        x0 = [-2, 1]
        # 对于多种设置下，使用最小二乘法计算 Rosenbrock 函数的最优化结果
        for jac, x_scale, tr_solver in product(
                ['2-point', '3-point', 'cs', jac_rosenbrock],
                [1.0, np.array([1.0, 0.2]), 'jac'],
                ['exact', 'lsmr']):
            with suppress_warnings() as sup:
                # 忽略警告，因为对于 'lm' 方法，"jac='(3-point|cs)'" 等同于 "jac='2-point'"
                sup.filter(
                    UserWarning,
                    "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'"
                )
                # 计算最优化结果
                res = least_squares(fun_rosenbrock, x0, jac, x_scale=x_scale,
                                    tr_solver=tr_solver, method=self.method)
            # 断言最优化结果 x 是否接近全局最优点 [1, 1]
            assert_allclose(res.x, x_opt)

    # 测试 Rosenbrock 函数的裁剪版本的优化结果是否接近全局最优点 [1, 1]
    def test_rosenbrock_cropped(self):
        x0 = [-2, 1]
        if self.method == 'lm':
            # 对于 'lm' 方法，应当引发 ValueError 异常
            assert_raises(ValueError, least_squares, fun_rosenbrock_cropped,
                          x0, method='lm')
        else:
            # 对于多种设置下，使用最小二乘法计算裁剪后的 Rosenbrock 函数的最优化结果
            for jac, x_scale, tr_solver in product(
                    ['2-point', '3-point', 'cs', jac_rosenbrock_cropped],
                    [1.0, np.array([1.0, 0.2]), 'jac'],
                    ['exact', 'lsmr']):
                # 计算最优化结果
                res = least_squares(
                    fun_rosenbrock_cropped, x0, jac, x_scale=x_scale,
                    tr_solver=tr_solver, method=self.method)
                # 断言最优化结果的 cost 是否接近 0，允许误差为 1e-14
                assert_allclose(res.cost, 0, atol=1e-14)

    # 测试输入参数维度错误的情况
    def test_fun_wrong_dimensions(self):
        # 应当引发 ValueError 异常，因为输入参数维度不正确
        assert_raises(ValueError, least_squares, fun_wrong_dimensions,
                      2.0, method=self.method)

    # 测试 Jacobian 函数维度错误的情况
    def test_jac_wrong_dimensions(self):
        # 应当引发 ValueError 异常，因为 Jacobian 函数维度不正确
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, jac_wrong_dimensions, method=self.method)

    # 测试函数与 Jacobian 函数维度不一致的情况
    def test_fun_and_jac_inconsistent_dimensions(self):
        x0 = [1, 2]
        # 应当引发 ValueError 异常，因为函数与 Jacobian 函数维度不一致
        assert_raises(ValueError, least_squares, fun_rosenbrock, x0,
                      jac_rosenbrock_bad_dim, method=self.method)

    # 测试初始点 x0 的维度错误的情况
    def test_x0_multidimensional(self):
        x0 = np.ones(4).reshape(2, 2)
        # 应当引发 ValueError 异常，因为初始点 x0 的维度不正确
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)
    # 定义测试函数，验证当初始值包含复数时，是否会引发值错误异常
    def test_x0_complex_scalar(self):
        # 设置一个复数初始值
        x0 = 2.0 + 0.0*1j
        # 断言调用最小二乘函数 `least_squares` 时会抛出值错误异常
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)

    # 定义测试函数，验证当初始值包含复数数组时，是否会引发值错误异常
    def test_x0_complex_array(self):
        # 设置包含复数的数组作为初始值
        x0 = [1.0, 2.0 + 0.0*1j]
        # 断言调用最小二乘函数 `least_squares` 时会抛出值错误异常
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)

    # 定义测试函数，验证边界值问题是否能正常收敛，特别处理了一个已知问题
    def test_bvp(self):
        # 这个测试是为了修复问题 #5556 而引入的。事实证明，dogbox 求解器在信任区域半径更新时存在错误，
        # 可能会阻塞其进展并导致无限循环。这个离散边界值问题就是触发该问题的一个示例。
        n = 10
        x0 = np.ones(n**2)
        # 根据所选方法设置最大迭代次数
        if self.method == 'lm':
            max_nfev = 5000  # 用于估计雅可比矩阵。
        else:
            max_nfev = 100
        # 调用最小二乘函数 `least_squares` 进行求解
        res = least_squares(fun_bvp, x0, ftol=1e-2, method=self.method,
                            max_nfev=max_nfev)
        # 断言实际函数评估次数小于最大迭代次数
        assert_(res.nfev < max_nfev)
        # 断言最终函数值的损失函数小于0.5
        assert_(res.cost < 0.5)

    # 定义测试函数，验证当所有容差都低于 eps 时是否会引发值错误异常
    def test_error_raised_when_all_tolerances_below_eps(self):
        # 测试所有容差都设置为0时是否会引发值错误异常
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      method=self.method, ftol=None, xtol=None, gtol=None)

    # 定义测试函数，验证只启用一个容差时是否能正常收敛
    def test_convergence_with_only_one_tolerance_enabled(self):
        # 对于 LM 方法，直接返回，不进行测试
        if self.method == 'lm':
            return
        x0 = [-2, 1]
        x_opt = [1, 1]
        # 对每一种单一容差的情况进行测试
        for ftol, xtol, gtol in [(1e-8, None, None),
                                  (None, 1e-8, None),
                                  (None, None, 1e-8)]:
            # 调用最小二乘函数 `least_squares` 进行求解
            res = least_squares(fun_rosenbrock, x0, jac=jac_rosenbrock,
                                ftol=ftol, gtol=gtol, xtol=xtol,
                                method=self.method)
            # 断言结果 `res.x` 收敛到预期的最优解 `x_opt`
            assert_allclose(res.x, x_opt)
class BoundsMixin:
    # 检查在给定不一致边界条件时是否引发 ValueError 异常
    def test_inconsistent(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(10.0, 0.0), method=self.method)

    # 检查在给定不可行边界条件时是否引发 ValueError 异常
    def test_infeasible(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(3., 4), method=self.method)

    # 检查在给定错误数量的边界条件时是否引发 ValueError 异常
    def test_wrong_number(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.,
                      bounds=(1., 2, 3), method=self.method)

    # 检查在给定不一致形状的边界条件时是否引发 ValueError 异常
    def test_inconsistent_shape(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(1.0, [2.0, 3.0]), method=self.method)
        # 一维数组不会被广播
        assert_raises(ValueError, least_squares, fun_rosenbrock, [1.0, 2.0],
                      bounds=([0.0], [3.0, 4.0]), method=self.method)

    # 检查在有效边界内求解时的结果是否符合预期
    def test_in_bounds(self):
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            # 在边界 (-1.0, 3.0) 内求解，检查结果是否接近 0.0
            res = least_squares(fun_trivial, 2.0, jac=jac,
                                bounds=(-1.0, 3.0), method=self.method)
            assert_allclose(res.x, 0.0, atol=1e-4)
            assert_equal(res.active_mask, [0])
            assert_(-1 <= res.x <= 3)
            # 在边界 (0.5, 3.0) 内求解，检查结果是否接近 0.5
            res = least_squares(fun_trivial, 2.0, jac=jac,
                                bounds=(0.5, 3.0), method=self.method)
            assert_allclose(res.x, 0.5, atol=1e-4)
            assert_equal(res.active_mask, [-1])
            assert_(0.5 <= res.x <= 3)

    # 检查在二维情况下边界条件的形状是否正确
    def test_bounds_shape(self):
        def get_bounds_direct(lb, ub):
            return lb, ub

        def get_bounds_instances(lb, ub):
            return Bounds(lb, ub)

        for jac in ['2-point', '3-point', 'cs', jac_2d_trivial]:
            for bounds_func in [get_bounds_direct, get_bounds_instances]:
                x0 = [1.0, 1.0]
                # 求解二维问题，边界条件由 bounds_func 生成
                res = least_squares(fun_2d_trivial, x0, jac=jac)
                assert_allclose(res.x, [0.0, 0.0])
                # 使用不同的边界条件测试求解结果是否符合预期
                res = least_squares(fun_2d_trivial, x0, jac=jac,
                                    bounds=bounds_func(0.5, [2.0, 2.0]),
                                    method=self.method)
                assert_allclose(res.x, [0.5, 0.5])
                res = least_squares(fun_2d_trivial, x0, jac=jac,
                                    bounds=bounds_func([0.3, 0.2], 3.0),
                                    method=self.method)
                assert_allclose(res.x, [0.3, 0.2])
                res = least_squares(
                    fun_2d_trivial, x0, jac=jac,
                    bounds=bounds_func([-1, 0.5], [1.0, 3.0]),
                    method=self.method)
                assert_allclose(res.x, [0.0, 0.5], atol=1e-5)
    # 定义一个测试方法，用于测试最小二乘法函数在不同边界条件下的表现
    def test_bounds_instances(self):
        # 对一个简单的函数进行最小二乘法拟合，不设定任何边界
        res = least_squares(fun_trivial, 0.5, bounds=Bounds())
        # 断言拟合结果的最优解接近于0.0，允许误差为1e-4
        assert_allclose(res.x, 0.0, atol=1e-4)

        # 对同一个简单函数进行最小二乘法拟合，设定下界为1.0
        res = least_squares(fun_trivial, 3.0, bounds=Bounds(lb=1.0))
        # 断言拟合结果的最优解接近于1.0，允许误差为1e-4
        assert_allclose(res.x, 1.0, atol=1e-4)

        # 对简单函数进行最小二乘法拟合，设定下界为-1.0，上界为1.0
        res = least_squares(fun_trivial, 0.5, bounds=Bounds(lb=-1.0, ub=1.0))
        # 断言拟合结果的最优解接近于0.0，允许误差为1e-4
        assert_allclose(res.x, 0.0, atol=1e-4)

        # 对简单函数进行最小二乘法拟合，设定上界为-1.0
        res = least_squares(fun_trivial, -3.0, bounds=Bounds(ub=-1.0))
        # 断言拟合结果的最优解接近于-1.0，允许误差为1e-4
        assert_allclose(res.x, -1.0, atol=1e-4)

        # 对二维简单函数进行最小二乘法拟合，设定下界为[-1.0, -1.0]，上界为1.0
        res = least_squares(fun_2d_trivial, [0.5, 0.5],
                            bounds=Bounds(lb=[-1.0, -1.0], ub=1.0))
        # 断言拟合结果的最优解接近于[0.0, 0.0]，允许误差为1e-5
        assert_allclose(res.x, [0.0, 0.0], atol=1e-5)

        # 对二维简单函数进行最小二乘法拟合，设定下界为[0.1, 0.1]
        res = least_squares(fun_2d_trivial, [0.5, 0.5],
                            bounds=Bounds(lb=[0.1, 0.1]))
        # 断言拟合结果的最优解接近于[0.1, 0.1]，允许误差为1e-5
        assert_allclose(res.x, [0.1, 0.1], atol=1e-5)

    # 使用pytest标记为“fail_slow(10)”的测试方法，用于测试在Rosenbrock函数下的边界条件
    @pytest.mark.fail_slow(10)
    def test_rosenbrock_bounds(self):
        # 定义多个初始点和对应的边界条件问题
        x0_1 = np.array([-2.0, 1.0])
        x0_2 = np.array([2.0, 2.0])
        x0_3 = np.array([-2.0, 2.0])
        x0_4 = np.array([0.0, 2.0])
        x0_5 = np.array([-1.2, 1.0])
        problems = [
            (x0_1, ([-np.inf, -1.5], np.inf)),
            (x0_2, ([-np.inf, 1.5], np.inf)),
            (x0_3, ([-np.inf, 1.5], np.inf)),
            (x0_4, ([-np.inf, 1.5], [1.0, np.inf])),
            (x0_2, ([1.0, 1.5], [3.0, 3.0])),
            (x0_5, ([-50.0, 0.0], [0.5, 100]))
        ]
        # 遍历每个初始点和边界条件组合
        for x0, bounds in problems:
            # 遍历每种雅可比矩阵计算方式、x尺度、追踪器求解器的组合
            for jac, x_scale, tr_solver in product(
                    ['2-point', '3-point', 'cs', jac_rosenbrock],
                    [1.0, [1.0, 0.5], 'jac'],
                    ['exact', 'lsmr']):
                # 使用最小二乘法拟合Rosenbrock函数，给定初始点、雅可比矩阵计算方式、边界条件等参数
                res = least_squares(fun_rosenbrock, x0, jac, bounds,
                                    x_scale=x_scale, tr_solver=tr_solver,
                                    method=self.method)
                # 断言拟合结果的最优性接近于0.0，允许误差为1e-5
                assert_allclose(res.optimality, 0.0, atol=1e-5)
# 定义一个稀疏混合类，用于测试精确的 TR 求解器功能
class SparseMixin:
    
    # 测试精确 TR 求解器是否能正确抛出 ValueError 异常
    def test_exact_tr_solver(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='exact', method=self.method)
        # 测试精确 TR 求解器在指定 jac_sparsity 参数时是否能正确抛出 ValueError 异常
        assert_raises(ValueError, least_squares, p.fun, p.x0,
                      tr_solver='exact', jac_sparsity=p.sparsity,
                      method=self.method)

    # 测试稀疏和稠密模式下的等价性
    def test_equivalence(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        # 使用 least_squares 函数分别对稀疏和稠密模式进行求解
        res_sparse = least_squares(
            sparse.fun, sparse.x0, jac=sparse.jac,
            method=self.method)
        res_dense = least_squares(
            dense.fun, dense.x0, jac=sparse.jac,
            method=self.method)
        # 断言稀疏和稠密模式下的函数调用次数相同
        assert_equal(res_sparse.nfev, res_dense.nfev)
        # 断言稀疏和稠密模式下的解向量 x 的近似值在给定的公差下相等
        assert_allclose(res_sparse.x, res_dense.x, atol=1e-20)
        # 断言稀疏模式下的成本函数近似为 0
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        # 断言稠密模式下的成本函数近似为 0
        assert_allclose(res_dense.cost, 0, atol=1e-20)

    # 测试 TR 求解器的选项设置
    def test_tr_options(self):
        p = BroydenTridiagonal()
        # 使用 least_squares 函数，传入 tr_options 参数进行求解
        res = least_squares(p.fun, p.x0, p.jac, method=self.method,
                            tr_options={'btol': 1e-10})
        # 断言解的成本函数近似为 0
        assert_allclose(res.cost, 0, atol=1e-20)

    # 测试错误的参数传递是否能正确抛出异常
    def test_wrong_parameters(self):
        p = BroydenTridiagonal()
        # 测试 tr_solver 参数为 'best' 时是否能正确抛出 ValueError 异常
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='best', method=self.method)
        # 测试 tr_solver 参数为 'lsmr' 且 tr_options 参数中包含 'tol' 键时是否能正确抛出 TypeError 异常
        assert_raises(TypeError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='lsmr', tr_options={'tol': 1e-10})

    # 测试求解器的选择，稀疏模式下的解应为稀疏矩阵，稠密模式下的解应为 numpy 数组
    def test_solver_selection(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        # 使用 least_squares 函数分别对稀疏和稠密模式进行求解
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac,
                                   method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac,
                                  method=self.method)
        # 断言稀疏模式下的成本函数近似为 0
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        # 断言稠密模式下的成本函数近似为 0
        assert_allclose(res_dense.cost, 0, atol=1e-20)
        # 断言稀疏模式下的雅可比矩阵为稀疏矩阵
        assert_(issparse(res_sparse.jac))
        # 断言稠密模式下的雅可比矩阵为 numpy 数组
        assert_(isinstance(res_dense.jac, np.ndarray))

    # 测试数值雅可比近似方法
    def test_numerical_jac(self):
        p = BroydenTridiagonal()
        # 遍历三种数值雅可比近似方法
        for jac in ['2-point', '3-point', 'cs']:
            # 使用 least_squares 函数分别对稀疏和稠密模式进行求解
            res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
            res_sparse = least_squares(
                p.fun, p.x0, jac, method=self.method,
                jac_sparsity=p.sparsity)
            # 断言稀疏和稠密模式下的函数调用次数相同
            assert_equal(res_dense.nfev, res_sparse.nfev)
            # 断言稀疏和稠密模式下的解向量 x 的近似值在给定的公差下相等
            assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
            # 断言稀疏模式下的成本函数近似为 0
            assert_allclose(res_sparse.cost, 0, atol=1e-20)
            # 断言稠密模式下的成本函数近似为 0
            assert_allclose(res_dense.cost, 0, atol=1e-20)

    # 标记测试为失败慢速测试，最多允许 10 次失败
    @pytest.mark.fail_slow(10)
    # 定义测试函数，用于测试 BroydenTridiagonal 类的边界情况
    def test_with_bounds(self):
        # 创建 BroydenTridiagonal 类的实例
        p = BroydenTridiagonal()
        # 使用 product 函数生成两个迭代器的笛卡尔积，其中一个包含 p.jac 和几种字符串，另一个包含 None 和 p.sparsity
        for jac, jac_sparsity in product(
                [p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
            # 调用 least_squares 函数进行最小二乘法优化，设置上界为 (p.lb, np.inf)，使用指定的 Jacobi 矩阵或字符串作为梯度计算方法
            res_1 = least_squares(
                p.fun, p.x0, jac, bounds=(p.lb, np.inf),
                method=self.method, jac_sparsity=jac_sparsity)
            # 调用 least_squares 函数进行最小二乘法优化，设置下界为 (-np.inf, p.ub)，使用指定的 Jacobi 矩阵或字符串作为梯度计算方法
            res_2 = least_squares(
                p.fun, p.x0, jac, bounds=(-np.inf, p.ub),
                method=self.method, jac_sparsity=jac_sparsity)
            # 调用 least_squares 函数进行最小二乘法优化，设置上界为 (p.lb, p.ub)，使用指定的 Jacobi 矩阵或字符串作为梯度计算方法
            res_3 = least_squares(
                p.fun, p.x0, jac, bounds=(p.lb, p.ub),
                method=self.method, jac_sparsity=jac_sparsity)
            # 断言优化结果的最优性满足给定的容差要求
            assert_allclose(res_1.optimality, 0, atol=1e-10)
            assert_allclose(res_2.optimality, 0, atol=1e-10)
            assert_allclose(res_3.optimality, 0, atol=1e-10)

    # 定义测试函数，用于测试错误的 Jacobi 稀疏性设置
    def test_wrong_jac_sparsity(self):
        # 创建 BroydenTridiagonal 类的实例
        p = BroydenTridiagonal()
        # 生成一个不正确的 Jacobi 稀疏性设置，期望引发 ValueError 异常
        sparsity = p.sparsity[:-1]
        assert_raises(ValueError, least_squares, p.fun, p.x0,
                      jac_sparsity=sparsity, method=self.method)

    # 定义测试函数，用于测试线性算子模式下的最小二乘法优化
    def test_linear_operator(self):
        # 创建 BroydenTridiagonal 类的实例，设置模式为 'operator'
        p = BroydenTridiagonal(mode='operator')
        # 调用 least_squares 函数进行最小二乘法优化，期望结果的 cost 接近 0
        res = least_squares(p.fun, p.x0, p.jac, method=self.method)
        assert_allclose(res.cost, 0.0, atol=1e-20)
        # 断言使用 'exact' 作为 tr_solver 参数调用 least_squares 函数时会引发 ValueError 异常
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method=self.method, tr_solver='exact')

    # 定义测试函数，用于测试 x_scale 和 jac_scale 参数设置
    def test_x_scale_jac_scale(self):
        # 创建 BroydenTridiagonal 类的实例
        p = BroydenTridiagonal()
        # 调用 least_squares 函数进行最小二乘法优化，设置 x_scale='jac'，期望结果的 cost 接近 0
        res = least_squares(p.fun, p.x0, p.jac, method=self.method,
                            x_scale='jac')
        assert_allclose(res.cost, 0.0, atol=1e-20)

        # 创建 BroydenTridiagonal 类的实例，设置模式为 'operator'
        p = BroydenTridiagonal(mode='operator')
        # 断言在模式为 'operator' 时，使用 x_scale='jac' 调用 least_squares 函数会引发 ValueError 异常
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method=self.method, x_scale='jac')
class LossFunctionMixin:
    # 测试不同损失函数的选项
    def test_options(self):
        for loss in LOSSES:
            # 使用最小二乘法 least_squares 进行测试，期望结果接近 0
            res = least_squares(fun_trivial, 2.0, loss=loss,
                                method=self.method)
            assert_allclose(res.x, 0, atol=1e-15)

        # 测试损失函数为 'hinge' 时应引发 ValueError 异常
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      loss='hinge', method=self.method)

    # 测试损失函数对应的函数值是否与实际残差相关，而非由损失函数影响修改
    def test_fun(self):
        for loss in LOSSES:
            # 使用最小二乘法 least_squares 进行测试
            res = least_squares(fun_trivial, 2.0, loss=loss,
                                method=self.method)
            # 断言结果的函数值 res.fun 等于原始函数 fun_trivial 的返回值
            assert_equal(res.fun, fun_trivial(res.x))

    # 测试损失函数在解决方案处的真实梯度
    def test_grad(self):
        x = np.array([2.0])  # res.x 将使用此值

        # 测试损失函数为 'linear' 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss='linear',
                            max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x * (x**2 + 5))

        # 测试损失函数为 'huber' 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
                            max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x)

        # 测试损失函数为 'soft_l1' 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad,
                        2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**0.5)

        # 测试损失函数为 'cauchy' 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2))

        # 测试损失函数为 'arctan' 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**4))

        # 测试自定义损失函数 cubic_soft_l1 时的梯度
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1,
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad,
                        2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**(2/3))
    # 定义一个测试方法，用于测试算法的鲁棒性
    def test_robustness(self):
        # 对于不同的噪声水平，执行以下循环
        for noise in [0.1, 1.0]:
            # 创建一个指数拟合问题实例，设置随机种子为0，以及指定的噪声水平
            p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)

            # 对于不同的雅可比矩阵计算方法，执行以下循环
            for jac in ['2-point', '3-point', 'cs', p.jac]:
                # 使用最小二乘法求解拟合问题，传入函数、初始猜测值、雅可比矩阵计算方法和优化方法
                res_lsq = least_squares(p.fun, p.p0, jac=jac,
                                        method=self.method)
                # 断言优化结果的最优性接近于0，允许的误差为1e-2
                assert_allclose(res_lsq.optimality, 0, atol=1e-2)
                # 对于定义的损失函数列表中的每一种损失函数，执行以下循环
                for loss in LOSSES:
                    # 如果损失函数是'linear'，则跳过本次循环
                    if loss == 'linear':
                        continue
                    # 使用带有指定损失函数和尺度参数的鲁棒最小二乘法求解拟合问题
                    res_robust = least_squares(
                        p.fun, p.p0, jac=jac, loss=loss, f_scale=noise,
                        method=self.method)
                    # 断言鲁棒优化结果的最优性接近于0，允许的误差为1e-2
                    assert_allclose(res_robust.optimality, 0, atol=1e-2)
                    # 断言鲁棒优化结果相对于最小二乘法结果更接近真实参数值
                    assert_(norm(res_robust.x - p.p_opt) <
                            norm(res_lsq.x - p.p_opt))
class TestDogbox(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    # 定义测试类 TestDogbox，继承了多个 mixin 类
    method = 'dogbox'  # 设置 method 属性为 'dogbox'


class TestTRF(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    # 定义测试类 TestTRF，继承了多个 mixin 类
    method = 'trf'  # 设置 method 属性为 'trf'

    def test_lsmr_regularization(self):
        # 定义测试方法 test_lsmr_regularization
        p = BroydenTridiagonal()  # 创建 BroydenTridiagonal 实例 p
        for regularize in [True, False]:  # 遍历 regularize 参数为 True 和 False 的情况
            res = least_squares(p.fun, p.x0, p.jac, method='trf',
                                tr_options={'regularize': regularize})
            # 使用 least_squares 函数进行最小二乘法优化，传入参数和选项
            assert_allclose(res.cost, 0, atol=1e-20)  # 断言优化结果的 cost 接近 0


class TestLM(BaseMixin):
    # 定义测试类 TestLM，继承了 BaseMixin 类
    method = 'lm'  # 设置 method 属性为 'lm'

    def test_bounds_not_supported(self):
        # 定义测试方法 test_bounds_not_supported
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, bounds=(-3.0, 3.0), method='lm')
        # 断言当使用 bounds 参数时会抛出 ValueError 异常

    def test_m_less_n_not_supported(self):
        # 定义测试方法 test_m_less_n_not_supported
        x0 = [-2, 1]  # 定义初始点 x0
        assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0,
                      method='lm')
        # 断言当 m 小于 n 时会抛出 ValueError 异常

    def test_sparse_not_supported(self):
        # 定义测试方法 test_sparse_not_supported
        p = BroydenTridiagonal()  # 创建 BroydenTridiagonal 实例 p
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method='lm')
        # 断言当使用稀疏矩阵时会抛出 ValueError 异常

    def test_jac_sparsity_not_supported(self):
        # 定义测试方法 test_jac_sparsity_not_supported
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      jac_sparsity=[1], method='lm')
        # 断言当使用 jac_sparsity 参数时会抛出 ValueError 异常

    def test_LinearOperator_not_supported(self):
        # 定义测试方法 test_LinearOperator_not_supported
        p = BroydenTridiagonal(mode="operator")  # 创建 BroydenTridiagonal 实例 p
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method='lm')
        # 断言当使用 LinearOperator 时会抛出 ValueError 异常

    def test_loss(self):
        # 定义测试方法 test_loss
        res = least_squares(fun_trivial, 2.0, loss='linear', method='lm')
        # 使用 loss='linear' 进行最小二乘法优化
        assert_allclose(res.x, 0.0, atol=1e-4)
        # 断言优化结果的 x 接近 0.0

        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      method='lm', loss='huber')
        # 断言当使用 loss='huber' 时会抛出 ValueError 异常


def test_basic():
    # 定义基本测试函数 test_basic
    # 测试 'method' 参数确实是可选的情况
    res = least_squares(fun_trivial, 2.0)
    # 调用 least_squares 函数，不传入 method 参数
    assert_allclose(res.x, 0, atol=1e-10)
    # 断言优化结果的 x 接近 0，精度为 1e-10


def test_small_tolerances_for_lm():
    # 定义测试函数 test_small_tolerances_for_lm
    for ftol, xtol, gtol in [(None, 1e-13, 1e-13),
                             (1e-13, None, 1e-13),
                             (1e-13, 1e-13, None)]:
        # 遍历 ftol, xtol, gtol 参数的组合
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, xtol=xtol,
                      ftol=ftol, gtol=gtol, method='lm')
        # 断言当设置 lm 方法的公差参数时会抛出 ValueError 异常


def test_fp32_gh12991():
    # 定义测试函数 test_fp32_gh12991
    # 检查在 least_squares 中可以使用较小的浮点数精度
    # 这是报告的最小工作示例 gh12991
    np.random.seed(1)

    x = np.linspace(0, 1, 100).astype("float32")
    y = np.random.random(100).astype("float32")

    def func(p, x):
        return p[0] + p[1] * x

    def err(p, x, y):
        return func(p, x) - y

    res = least_squares(err, [-1.0, -1.0], args=(x, y))
    # 使用最小二乘法优化错误函数 err，传入参数和参数数组
    # 之前对于这个问题，初始的雅可比矩阵计算结果会全部为 0
    # 优化会立即终止，nfev=1，报告成功的最小化（实际上不应该）
    # 但与初始解相比没有改变。
    # 这是因为近似导数的问题
    # 确保优化结果的评估函数调用次数大于2
    assert res.nfev > 2
    # 使用 assert_allclose 函数验证优化结果的参数 x 是否接近给定的参考值数组，设置容差为 5e-5
    assert_allclose(res.x, np.array([0.4082241, 0.15530563]), atol=5e-5)
def test_gh_18793_and_19351():
    answer = 1e-12
    initial_guess = 1.1e-12

    # 定义一个计算 chi-squared 的函数，用于最小化
    def chi2(x):
        return (x-answer)**2

    gtol = 1e-15
    # 使用最小二乘法进行优化，使用 'trf' 算法，设置初始值和梯度容许误差
    res = least_squares(chi2, x0=initial_guess, gtol=1e-15, bounds=(0, np.inf))

    # Original motivation: gh-18793
    # 如果我们选择一个接近解的初始条件，我们不应该返回一个离解更远的答案

    # Update: gh-19351
    # 然而，这个要求与 'trf' 算法的逻辑不太匹配。
    # 一些问题报告表明，在假定的修复后，出现了一些回归。
    # 只要满足收敛条件，返回的解就是好的。
    # 特别是在这种情况下，缩放后的梯度应该足够低。

    # 计算缩放向量和缩放后的梯度向量
    scaling, _ = CL_scaling_vector(res.x, res.grad,
                                   np.atleast_1d(0), np.atleast_1d(np.inf))
    
    # 断言优化是否通过梯度收敛
    assert res.status == 1  # Converged by gradient
    # 断言缩放后的梯度的无穷范数是否小于容许误差 gtol
    assert np.linalg.norm(res.grad * scaling, ord=np.inf) < gtol


def test_gh_19103():
    # 检查 least_squares 使用 'trf' 方法时选择严格可行点，
    # 因此成功而不是失败，
    # 当初始猜测恰好在边界点报告时。
    # 这是从 gh191303 中简化的例子

    ydata = np.array([0.] * 66 + [
        1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
        1., 1., 1., 0., 0., 0., 1., 0., 0., 2., 1.,
        0., 3., 1., 6., 5., 0., 0., 2., 8., 4., 4.,
        6., 9., 7., 2., 7., 8., 2., 13., 9., 8., 11.,
        10., 13., 14., 19., 11., 15., 18., 26., 19., 32., 29.,
        28., 36., 32., 35., 36., 43., 52., 32., 58., 56., 52.,
        67., 53., 72., 88., 77., 95., 94., 84., 86., 101., 107.,
        108., 118., 96., 115., 138., 137.,
    ])
    xdata = np.arange(0, ydata.size) * 0.1

    # 定义指数函数的残差函数
    def exponential_wrapped(params):
        A, B, x0 = params
        return A * np.exp(B * (xdata - x0)) - ydata

    x0 = [0.01, 1., 5.]
    bounds = ((0.01, 0, 0), (np.inf, 10, 20.9))
    # 使用 'trf' 方法的 least_squares 函数进行优化
    res = least_squares(exponential_wrapped, x0, method='trf', bounds=bounds)
    # 断言优化是否成功
    assert res.success
```