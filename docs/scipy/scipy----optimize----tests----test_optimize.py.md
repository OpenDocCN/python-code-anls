# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_optimize.py`

```
"""
Unit tests for optimization routines from optimize.py

Authors:
   Ed Schofield, Nov 2005
   Andrew Straw, April 2008

To run it in its simplest form::
  nosetests test_optimize.py

"""
# 导入必要的模块和库
import itertools                 # 导入 itertools 模块，用于迭代操作
import platform                  # 导入 platform 模块，用于访问底层操作系统信息
import numpy as np               # 导入 NumPy 库，并用 np 别名引用
from numpy.testing import (     # 导入 NumPy 测试模块中的各种断言函数
    assert_allclose, assert_equal, assert_almost_equal,
    assert_no_warnings, assert_warns, assert_array_less, suppress_warnings
)
import pytest                    # 导入 Pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数并重命名为 assert_raises

from scipy import optimize       # 导入 SciPy 中的 optimize 模块
from scipy.optimize._minimize import Bounds, NonlinearConstraint  # 导入 minimize 子模块中的特定类和函数
from scipy.optimize._minimize import (  # 导入 minimize 子模块中的常量和函数列表
    MINIMIZE_METHODS, MINIMIZE_METHODS_NEW_CB, MINIMIZE_SCALAR_METHODS
)
from scipy.optimize._linprog import LINPROG_METHODS  # 导入 linprog 子模块中的线性规划方法列表
from scipy.optimize._root import ROOT_METHODS         # 导入 root 子模块中的根查找方法列表
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS  # 导入 root_scalar 子模块中的根查找方法列表
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS  # 导入 qap 子模块中的二次分配方法列表
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS  # 导入 differentiable_functions 子模块中的特定类和函数
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult  # 导入 optimize 子模块中的特定类和函数
from scipy.optimize import rosen, rosen_der, rosen_hess  # 导入 optimize 中的特定函数

from scipy.sparse import (       # 导入 SciPy 稀疏矩阵模块中的特定类和函数
    coo_matrix, csc_matrix, csr_matrix, coo_array,
    csr_array, csc_array
)

def test_check_grad():
    # Verify if check_grad is able to estimate the derivative of the
    # expit (logistic sigmoid) function.

    def expit(x):
        return 1 / (1 + np.exp(-x))  # 定义 logistic sigmoid 函数

    def der_expit(x):
        return np.exp(-x) / (1 + np.exp(-x))**2  # 定义 logistic sigmoid 函数的导数

    x0 = np.array([1.5])  # 初始化输入值 x0

    r = optimize.check_grad(expit, der_expit, x0)  # 使用 optimize 模块中的 check_grad 函数计算估计导数
    assert_almost_equal(r, 0)  # 断言估计导数与真实导数接近

    r = optimize.check_grad(expit, der_expit, x0,
                            direction='random', seed=1234)  # 使用随机方向和种子计算估计导数
    assert_almost_equal(r, 0)  # 断言估计导数与真实导数接近

    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-6)  # 使用指定的 epsilon 计算估计导数
    assert_almost_equal(r, 0)  # 断言估计导数与真实导数接近

    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-6,
                            direction='random', seed=1234)  # 使用指定的 epsilon、随机方向和种子计算估计导数
    assert_almost_equal(r, 0)  # 断言估计导数与真实导数接近

    # Check if the epsilon parameter is being considered.
    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=1e-1) - 0)  # 检查 epsilon 参数是否起作用
    assert r > 1e-7  # 断言结果大于给定的阈值

    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=1e-1,
                                direction='random', seed=1234) - 0)  # 使用指定的 epsilon、随机方向和种子计算估计导数
    assert r > 1e-7  # 断言结果大于给定的阈值

    def x_sinx(x):
        return (x * np.sin(x)).sum()  # 定义函数 x * sin(x) 的和

    def der_x_sinx(x):
        return np.sin(x) + x * np.cos(x)  # 定义函数 x * sin(x) 的导数

    x0 = np.arange(0, 2, 0.2)  # 初始化输入值 x0

    r = optimize.check_grad(x_sinx, der_x_sinx, x0,
                            direction='random', seed=1234)  # 使用随机方向和种子计算估计导数
    assert_almost_equal(r, 0)  # 断言估计导数与真实导数接近

    assert_raises(ValueError, optimize.check_grad,
                  x_sinx, der_x_sinx, x0,
                  direction='random_projection', seed=1234)  # 断言 ValueError 异常被抛出

    # checking can be done for derivatives of vector valued functions
    # 使用 optimize 模块中的 check_grad 函数来检查 Himmelblau 函数的梯度和黑塞矩阵计算是否正确
    # himmelblau_grad 是梯度函数，himmelblau_hess 是黑塞矩阵函数，himmelblau_x0 是起始点
    # direction='all' 表示检查所有方向的梯度
    # seed=1234 是随机种子，用于确定性地生成随机数
    r = optimize.check_grad(himmelblau_grad, himmelblau_hess, himmelblau_x0,
                            direction='all', seed=1234)
    
    # 断言检查优化后的梯度 r 是否小于 5e-7，如果不小于则会触发 AssertionError
    assert r < 5e-7
class CheckOptimize:
    """ Base test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    # 设置测试方法的初始化
    def setup_method(self):
        # 定义 F 矩阵，表示约束矩阵
        self.F = np.array([[1, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [1, 0, 0],
                           [1, 0, 0]])
        # 定义 K 向量，表示约束条件
        self.K = np.array([1., 0.3, 0.5])
        # 设置初始参数为零向量
        self.startparams = np.zeros(3, np.float64)
        # 设定期望的最优解
        self.solution = np.array([0., -0.524869316, 0.487525860])
        # 设定最大迭代次数
        self.maxiter = 1000
        # 函数调用次数初始化
        self.funccalls = 0
        # 梯度调用次数初始化
        self.gradcalls = 0
        # 追踪变量的空列表
        self.trace = []

    # 定义目标函数 func
    def func(self, x):
        # 增加函数调用次数计数器
        self.funccalls += 1
        # 如果迭代次数超过6000次，则引发运行时错误
        if self.funccalls > 6000:
            raise RuntimeError("too many iterations in optimization routine")
        # 计算 F 矩阵与参数 x 的内积
        log_pdot = np.dot(self.F, x)
        # 计算对数配分函数 logZ
        logZ = np.log(sum(np.exp(log_pdot)))
        # 计算目标函数值 f
        f = logZ - np.dot(self.K, x)
        # 将当前参数 x 的副本追加到追踪列表中
        self.trace.append(np.copy(x))
        # 返回目标函数值 f
        return f

    # 定义目标函数的梯度 grad
    def grad(self, x):
        # 增加梯度调用次数计数器
        self.gradcalls += 1
        # 计算 F 矩阵与参数 x 的内积
        log_pdot = np.dot(self.F, x)
        # 计算对数配分函数 logZ
        logZ = np.log(sum(np.exp(log_pdot)))
        # 计算概率向量 p
        p = np.exp(log_pdot - logZ)
        # 返回目标函数的梯度值
        return np.dot(self.F.transpose(), p) - self.K

    # 定义目标函数的黑塞矩阵 hess
    def hess(self, x):
        # 计算 F 矩阵与参数 x 的内积
        log_pdot = np.dot(self.F, x)
        # 计算对数配分函数 logZ
        logZ = np.log(sum(np.exp(log_pdot)))
        # 计算概率向量 p
        p = np.exp(log_pdot - logZ)
        # 计算黑塞矩阵
        return np.dot(self.F.T,
                      np.dot(np.diag(p), self.F - np.dot(self.F.T, p)))

    # 定义黑塞矩阵与向量 p 的乘积 hessp
    def hessp(self, x, p):
        # 计算黑塞矩阵与向量 p 的乘积
        return np.dot(self.hess(x), p)


class CheckOptimizeParameterized(CheckOptimize):
    pass
    def test_cg(self):
        # conjugate gradient optimization routine
        # 如果使用包装器，设置优化选项并调用 optimize.minimize 函数
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='CG', jac=self.grad,
                                    options=opts)
            # 提取优化结果中的参数、目标函数值、函数调用次数、梯度调用次数和警告标志
            params, fopt, func_calls, grad_calls, warnflag = \
                res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        else:
            # 使用 optimize.fmin_cg 函数进行优化
            retval = optimize.fmin_cg(self.func, self.startparams,
                                      self.grad, (), maxiter=self.maxiter,
                                      full_output=True, disp=self.disp,
                                      retall=False)
            # 解析 optimize.fmin_cg 返回的结果
            (params, fopt, func_calls, grad_calls, warnflag) = retval

        # 断言优化后的参数得到的函数值与已知解的函数值非常接近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 确保函数调用次数和梯度调用次数与预期值相符，这些数值来自 SciPy 0.7.0
        assert self.funccalls == 9, self.funccalls
        assert self.gradcalls == 7, self.gradcalls

        # 确保函数的行为与预期一致，这些数值来自 SciPy 0.7.0
        assert_allclose(self.trace[2:4],
                        [[0, -0.5, 0.5],
                         [0, -5.05700028e-01, 4.95985862e-01]],
                        atol=1e-14, rtol=1e-7)

    def test_cg_cornercase(self):
        # 定义一个测试函数 f(r)，用于测试边缘情况
        def f(r):
            return 2.5 * (1 - np.exp(-1.5*(r - 0.5)))**2

        # 对多个初始猜测值进行测试
        # 如果初始猜测值离最小值太远，函数可能会落入指数函数的平缓区域
        for x0 in np.linspace(-0.75, 3, 71):
            # 使用 optimize.minimize 函数进行优化，方法选择 'CG'
            sol = optimize.minimize(f, [x0], method='CG')
            # 断言优化成功
            assert sol.success
            # 断言优化后的参数与预期的最小值非常接近
            assert_allclose(sol.x, [0.5], rtol=1e-5)
    def test_bfgs(self):
        # Broyden-Fletcher-Goldfarb-Shanno optimization routine

        # 如果使用了包装器，设置优化选项
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 使用 BFGS 方法进行优化
            res = optimize.minimize(self.func, self.startparams,
                                    jac=self.grad, method='BFGS', args=(),
                                    options=opts)

            # 从优化结果中获取参数、最优函数值、最优梯度、最优逆 Hessian 矩阵、函数调用次数、梯度调用次数、警告标志
            params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = (
                    res['x'], res['fun'], res['jac'], res['hess_inv'],
                    res['nfev'], res['njev'], res['status'])
        else:
            # 使用 fmin_bfgs 函数进行优化
            retval = optimize.fmin_bfgs(self.func, self.startparams, self.grad,
                                        args=(), maxiter=self.maxiter,
                                        full_output=True, disp=self.disp,
                                        retall=False)
            # 从返回值中获取参数、最优函数值、最优梯度、最优逆 Hessian 矩阵、函数调用次数、梯度调用次数、警告标志
            (params, fopt, gopt, Hopt,
             func_calls, grad_calls, warnflag) = retval

        # 断言优化后的函数值与预期函数值非常接近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 确保函数调用次数与已知值相符；这些值来自 SciPy 0.7.0，不允许其增加
        assert self.funccalls == 10, self.funccalls
        # 确保梯度调用次数与已知值相符；这些值来自 SciPy 0.7.0，不允许其增加
        assert self.gradcalls == 8, self.gradcalls

        # 确保函数的行为与已知值相符；这些值来自 SciPy 0.7.0
        assert_allclose(self.trace[6:8],
                        [[0, -5.25060743e-01, 4.87748473e-01],
                         [0, -5.24885582e-01, 4.87530347e-01]],
                        atol=1e-14, rtol=1e-7)

    def test_bfgs_hess_inv0_neg(self):
        # Ensure that BFGS does not accept neg. def. initial inverse
        # Hessian estimate.

        # 使用 pytest 来确保当提供的初始逆 Hessian 估计是负定时，抛出 ValueError 异常
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't "
                           "positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            opts = {'disp': self.disp, 'hess_inv0': -np.eye(5)}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(),
                              options=opts)

    def test_bfgs_hess_inv0_semipos(self):
        # Ensure that BFGS does not accept semi pos. def. initial inverse
        # Hessian estimate.

        # 使用 pytest 来确保当提供的初始逆 Hessian 估计是半正定时，抛出 ValueError 异常
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't "
                           "positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            hess_inv0 = np.eye(5)
            hess_inv0[0, 0] = 0
            opts = {'disp': self.disp, 'hess_inv0': hess_inv0}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(),
                              options=opts)
    def test_bfgs_hess_inv0_sanity(self):
        # 确保 BFGS 正确处理 `hess_inv0` 参数。
        # 定义被优化函数为 Rosenbrock 函数
        fun = optimize.rosen
        # 初始点
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        # 选项字典，设置显示选项和 Hessian 逆矩阵的初始值
        opts = {'disp': self.disp, 'hess_inv0': 1e-2 * np.eye(5)}
        # 使用 BFGS 方法进行优化
        res = optimize.minimize(fun, x0=x0, method='BFGS', args=(),
                                options=opts)
        # 作为对照，使用相同初始点但不设置 hess_inv0 参数进行优化
        res_true = optimize.minimize(fun, x0=x0, method='BFGS', args=(),
                                     options={'disp': self.disp})
        # 断言优化结果的函数值接近
        assert_allclose(res.fun, res_true.fun, atol=1e-6)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_bfgs_infinite(self):
        # 测试当最小值是负无穷大的极端情况。参见 gh-2019。
        # 定义测试函数
        def func(x):
            return -np.e ** (-x)
        # 定义测试函数的导数
        def fprime(x):
            return -func(x)
        # 初始点
        x0 = [0]
        # 忽略数值运算警告
        with np.errstate(over='ignore'):
            # 如果使用包装器
            if self.use_wrapper:
                # 设置选项字典，仅包含显示选项
                opts = {'disp': self.disp}
                # 使用 BFGS 方法进行优化
                x = optimize.minimize(func, x0, jac=fprime, method='BFGS',
                                      args=(), options=opts)['x']
            else:
                # 使用 fmin_bfgs 函数进行优化
                x = optimize.fmin_bfgs(func, x0, fprime, disp=self.disp)
            # 断言优化后的函数值不是有限的
            assert not np.isfinite(func(x))

    def test_bfgs_xrtol(self):
        # 测试 #17345 的 xrtol 参数
        # 初始点
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        # 使用 BFGS 方法进行优化，设置 xrtol 参数
        res = optimize.minimize(optimize.rosen,
                                x0, method='bfgs', options={'xrtol': 1e-3})
        # 使用 BFGS 方法进行优化，设置 gtol 参数作为对照
        ref = optimize.minimize(optimize.rosen,
                                x0, method='bfgs', options={'gtol': 1e-3})
        # 断言两次优化迭代次数不同
        assert res.nit != ref.nit

    def test_bfgs_c1(self):
        # 测试 #18977，当 c1 值过低导致在初始参数较差时精度丢失
        # 初始点
        x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
        # 使用 BFGS 方法进行优化，设置较小的 c1 参数
        res_c1_small = optimize.minimize(optimize.rosen,
                                         x0, method='bfgs', options={'c1': 1e-8})
        # 使用 BFGS 方法进行优化，设置较大的 c1 参数作为对照
        res_c1_big = optimize.minimize(optimize.rosen,
                                       x0, method='bfgs', options={'c1': 1e-1})
        # 断言使用较小 c1 参数时的函数评估次数多于使用较大 c1 参数时的次数
        assert res_c1_small.nfev > res_c1_big.nfev

    def test_bfgs_c2(self):
        # 测试修改 c2 参数会导致不同迭代次数
        # 初始点
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        # 使用默认 c2 参数进行 BFGS 优化
        res_default = optimize.minimize(optimize.rosen,
                                        x0, method='bfgs', options={'c2': .9})
        # 使用修改后的较小 c2 参数进行 BFGS 优化作为对照
        res_mod = optimize.minimize(optimize.rosen,
                                    x0, method='bfgs', options={'c2': 1e-2})
        # 断言使用默认 c2 参数时的迭代次数多于使用修改后 c2 参数时的次数
        assert res_default.nit > res_mod.nit

    @pytest.mark.parametrize(["c1", "c2"], [[0.5, 2],
                                            [-0.1, 0.1],
                                            [0.2, 0.1]])
    # 定义一个测试方法，用于检查参数 c1 和 c2 是否触发 ValueError 异常
    def test_invalid_c1_c2(self, c1, c2):
        # 使用 pytest 的断言来检查是否引发指定异常，并验证异常消息内容
        with pytest.raises(ValueError, match="'c1' and 'c2'"):
            # 初始化一个起始点列表 x0
            x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
            # 调用 optimize.minimize 函数进行优化，使用 'cg' 方法和指定的 c1、c2 参数
            optimize.minimize(optimize.rosen, x0, method='cg',
                              options={'c1': c1, 'c2': c2})

    # 定义 Powell 方法的测试方法
    def test_powell(self):
        # 如果使用了包装器，则设置优化选项字典 opts
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 使用 optimize.minimize 函数进行 Powell 方法的优化，并返回结果 res
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Powell', options=opts)
            # 从结果 res 中解包出优化后的参数、目标函数值等信息
            params, fopt, direc, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['direc'], res['nit'],
                    res['nfev'], res['status'])
        else:
            # 如果没有使用包装器，则直接调用 optimize.fmin_powell 函数
            retval = optimize.fmin_powell(self.func, self.startparams,
                                          args=(), maxiter=self.maxiter,
                                          full_output=True, disp=self.disp,
                                          retall=False)
            # 解包 optimize.fmin_powell 的返回值
            (params, fopt, direc, numiter, func_calls, warnflag) = retval

        # 使用 assert_allclose 检查优化后的参数是否接近预期解决方案的参数
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)
        # 检查优化后的除第一个参数外的其余参数是否与预期解决方案的参数接近
        assert_allclose(params[1:], self.solution[1:], atol=5e-6)

        # 确保函数调用计数是“已知良好”的，这些数据来自 SciPy 0.7.0，不允许它们增加
        #
        # 但是，必须增加一些余地：确切的评估计数对数值误差很敏感，
        # 并且浮点计算在不同机器上不是位对位可重现的，
        # 使用例如 MKL 时，数据对齐等会影响舍入误差。
        #
        assert self.funccalls <= 116 + 20, self.funccalls
        # 确保梯度调用次数为 0
        assert self.gradcalls == 0, self.gradcalls

    # 使用 pytest.mark.xfail 标记的测试，说明 test_powell 的某部分在某些平台上可能会失败，
    # 但是 Powell 方法返回的解仍然有效
    @pytest.mark.xfail(reason="This part of test_powell fails on some "
                       "platforms, but the solution returned by powell is "
                       "still valid.")
    # 定义名为 test_powell_gh14014 的测试方法，用于测试 Powell 方法优化的特定情况
    def test_powell_gh14014(self):
        # 此部分 test_powell 在某些 CI 平台上开始失败；参见 gh-14014。由于解决方案仍然正确，
        # 并且 test_powell 中的注释表明位中的小差异已知会改变解决方案的“迹”，所以安全地
        # 使用 xfail 来使 CI 绿色通过，稍后再进行调查。

        # Powell 方向集优化例程

        # 如果使用包装器
        if self.use_wrapper:
            # 设定优化选项字典
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 调用 optimize.minimize 执行 Powell 方法优化
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Powell', options=opts)
            # 从结果中获取参数、最优值、方向、迭代次数、函数调用次数、警告标志
            params, fopt, direc, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['direc'], res['nit'],
                    res['nfev'], res['status'])
        else:
            # 否则调用 optimize.fmin_powell 执行 Powell 方法优化
            retval = optimize.fmin_powell(self.func, self.startparams,
                                          args=(), maxiter=self.maxiter,
                                          full_output=True, disp=self.disp,
                                          retall=False)
            # 从返回值中获取参数、最优值、方向、迭代次数、函数调用次数、警告标志
            (params, fopt, direc, numiter, func_calls, warnflag) = retval

        # 确保函数的行为相同；这是来自 SciPy 0.7.0 的测试
        assert_allclose(self.trace[34:39],
                        [[0.72949016, -0.44156936, 0.47100962],
                         [0.72949016, -0.44156936, 0.48052496],
                         [1.45898031, -0.88313872, 0.95153458],
                         [0.72949016, -0.44156936, 0.47576729],
                         [1.72949016, -0.44156936, 0.47576729]],
                        atol=1e-14, rtol=1e-7)

    # 定义名为 test_powell_bounded 的测试方法，用于测试带边界条件的 Powell 方法优化
    def test_powell_bounded(self):
        # Powell 方向集优化例程
        # 与上面的 test_powell 相同，但是带有边界条件

        # 根据 self.startparams 中的参数定义边界条件
        bounds = [(-np.pi, np.pi) for _ in self.startparams]
        
        # 如果使用包装器
        if self.use_wrapper:
            # 设定优化选项字典
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 调用 optimize.minimize 执行带边界条件的 Powell 方法优化
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    bounds=bounds,
                                    method='Powell', options=opts)
            # 从结果中获取参数和函数调用次数
            params, func_calls = (res['x'], res['nfev'])

            # 断言函数调用次数与预期相符
            assert func_calls == self.funccalls
            # 断言优化后的函数值与预期的解决方案函数值非常接近
            assert_allclose(self.func(params), self.func(self.solution),
                            atol=1e-6, rtol=1e-5)

            # 精确的评估次数对数值误差敏感，在不同机器上浮点计算不是位对位可重现的，
            # 使用 MKL 等会影响舍入误差的数据对齐等因素。
            # 在我的机器上需要 155 次调用，但我们可以增加 +20 的余量，与 `test_powell` 中使用的一样
            assert self.funccalls <= 155 + 20
            # 梯度调用次数应为 0
            assert self.gradcalls == 0
    # 定义测试函数 test_neldermead，用于测试 Nelder-Mead 单纯形算法
    def test_neldermead(self):
        # 如果使用包装器
        if self.use_wrapper:
            # 设置优化选项
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 调用 optimize.minimize 函数进行优化，使用 Nelder-Mead 方法
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Nelder-mead', options=opts)
            # 从优化结果中获取参数、最优值、迭代次数、函数调用次数、警告标志
            params, fopt, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['nit'], res['nfev'],
                    res['status'])
        else:
            # 使用 optimize.fmin 函数进行优化，使用 Nelder-Mead 方法
            retval = optimize.fmin(self.func, self.startparams,
                                   args=(), maxiter=self.maxiter,
                                   full_output=True, disp=self.disp,
                                   retall=False)
            # 从优化结果中获取参数、最优值、迭代次数、函数调用次数、警告标志
            (params, fopt, numiter, func_calls, warnflag) = retval

        # 断言优化后的函数值与期望的解的函数值之间的接近程度在指定的公差范围内
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 确保函数调用次数与已知值相同，这些值来自 SciPy 0.7.0，不允许它们增加
        assert self.funccalls == 167, self.funccalls
        # 确保梯度计算次数为 0
        assert self.gradcalls == 0, self.gradcalls

        # 确保函数行为与已知值相同，这些值来自 SciPy 0.7.0
        assert_allclose(self.trace[76:78],
                        [[0.1928968, -0.62780447, 0.35166118],
                         [0.19572515, -0.63648426, 0.35838135]],
                        atol=1e-14, rtol=1e-7)
    # 定义测试函数，用于测试 Nelder-Mead 单纯形算法的初始单纯形生成
    def test_neldermead_initial_simplex(self):
        # 创建一个形状为 (4, 3) 的全零数组作为初始单纯形
        simplex = np.zeros((4, 3))
        # 将初始单纯形的值设置为起始参数 self.startparams
        simplex[...] = self.startparams
        # 修改初始单纯形的第 j 列，使得每个顶点稍微偏离
        for j in range(3):
            simplex[j+1, j] += 0.1

        # 根据是否使用包装器来选择不同的优化方法和参数设置
        if self.use_wrapper:
            # 配置优化参数 opts，包括最大迭代次数、是否显示优化过程、返回所有迭代结果、初始单纯形
            opts = {'maxiter': self.maxiter, 'disp': False,
                    'return_all': True, 'initial_simplex': simplex}
            # 使用 optimize.minimize 函数进行优化，采用 Nelder-Mead 方法
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Nelder-mead', options=opts)
            # 解析优化结果 res，获取优化后的参数、目标函数值、迭代次数、函数调用次数、警告标志等信息
            params, fopt, numiter, func_calls, warnflag = (res['x'],
                                                           res['fun'],
                                                           res['nit'],
                                                           res['nfev'],
                                                           res['status'])
            # 断言初始单纯形在优化过程中的第一个顶点与 res['allvecs'] 中记录的第一个顶点相近
            assert_allclose(res['allvecs'][0], simplex[0])
        else:
            # 使用 optimize.fmin 函数进行优化，采用 Nelder-Mead 方法
            retval = optimize.fmin(self.func, self.startparams,
                                   args=(), maxiter=self.maxiter,
                                   full_output=True, disp=False, retall=False,
                                   initial_simplex=simplex)
            # 解析优化结果 retval，获取优化后的参数、目标函数值、迭代次数、函数调用次数、警告标志等信息
            (params, fopt, numiter, func_calls, warnflag) = retval

        # 断言优化后的参数与预期解 self.solution 的函数值相近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 断言函数调用次数与预期值相符合
        assert self.funccalls == 100, self.funccalls
        # 断言梯度调用次数为 0
        assert self.gradcalls == 0, self.gradcalls

        # 断言函数在特定追踪点的输出与预期值相近
        # 这些预期值来自 SciPy 0.15.0
        assert_allclose(self.trace[50:52],
                        [[0.14687474, -0.5103282, 0.48252111],
                         [0.14474003, -0.5282084, 0.48743951]],
                        atol=1e-14, rtol=1e-7)
    def test_neldermead_initial_simplex_bad(self):
        # 检查当初始简单形状不良时是否会失败
        bad_simplices = []  # 初始化一个空列表，用于存储不良简单形状

        simplex = np.zeros((3, 2))  # 创建一个3x2的全零数组
        simplex[...] = self.startparams[:2]  # 将self.startparams的前两个元素复制到simplex中
        for j in range(2):
            simplex[j+1, j] += 0.1  # 对simplex的每一行进行微小调整
        bad_simplices.append(simplex)  # 将调整后的simplex添加到不良简单形状列表中

        simplex = np.zeros((3, 3))  # 创建一个3x3的全零数组
        bad_simplices.append(simplex)  # 将全零数组simplex添加到不良简单形状列表中

        for simplex in bad_simplices:  # 遍历不良简单形状列表中的每一个简单形状
            if self.use_wrapper:  # 如果使用了包装器
                opts = {'maxiter': self.maxiter, 'disp': False,
                        'return_all': False, 'initial_simplex': simplex}
                # 使用特定的初始简单形状进行优化，期望抛出值错误异常
                assert_raises(ValueError,
                              optimize.minimize,
                              self.func,
                              self.startparams,
                              args=(),
                              method='Nelder-mead',
                              options=opts)
            else:  # 如果没有使用包装器
                # 使用特定的初始简单形状进行优化，期望抛出值错误异常
                assert_raises(ValueError, optimize.fmin,
                              self.func, self.startparams,
                              args=(), maxiter=self.maxiter,
                              full_output=True, disp=False, retall=False,
                              initial_simplex=simplex)

    def test_neldermead_x0_ub(self):
        # 检查当x0 == ub时是否能正确进行最小化
        # gh19991
        def quad(x):
            return np.sum(x**2)

        # 对一个变量的情况进行测试
        res = optimize.minimize(
            quad,
            [1],
            bounds=[(0, 1.)],
            method='nelder-mead'
        )
        assert_allclose(res.x, [0])  # 断言最小化结果的x接近0

        # 对两个变量的情况进行测试
        res = optimize.minimize(
            quad,
            [1, 2],
            bounds=[(0, 1.), (1, 3.)],
            method='nelder-mead'
        )
        assert_allclose(res.x, [0, 1])  # 断言最小化结果的x接近[0, 1]

    def test_ncg_negative_maxiter(self):
        # gh-8241的回归测试
        opts = {'maxiter': -1}  # 设置最大迭代次数为负数
        result = optimize.minimize(self.func, self.startparams,
                                   method='Newton-CG', jac=self.grad,
                                   args=(), options=opts)
        assert result.status == 1  # 断言结果的状态为1（预期的失败状态）

    def test_ncg_zero_xtol(self):
        # gh-20214的回归测试
        def cosine(x):
            return np.cos(x[0])

        def jac(x):
            return -np.sin(x[0])

        x0 = [0.1]  # 设置初始点
        xtol = 0  # 设置xtol为0
        result = optimize.minimize(cosine,
                                   x0=x0,
                                   jac=jac,
                                   method="newton-cg",
                                   options=dict(xtol=xtol))
        assert result.status == 0  # 断言结果的状态为0（预期的成功状态）
        assert_almost_equal(result.x[0], np.pi)  # 断言最小化结果的x接近π
    # 定义一个名为 test_ncg 的测试方法
    def test_ncg(self):
        # 如果使用了包装器
        if self.use_wrapper:
            # 设置优化选项字典，包括最大迭代次数、是否显示优化过程，不返回所有信息
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            # 使用 Newton-CG 方法进行优化，使用给定的函数、起始参数、梯度函数和选项
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       args=(), options=opts)['x']
        else:
            # 使用 fmin_ncg 函数进行优化，使用给定的函数、起始参数、梯度函数和最大迭代次数等参数
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        # 将优化得到的参数赋给变量 params
        params = retval

        # 断言优化后的函数值与预期解的函数值非常接近，误差限为 1e-6
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 断言函数调用次数与预期值相符，这些值来自 SciPy 0.7.0 版本
        assert self.funccalls == 7, self.funccalls
        # 断言梯度函数调用次数不超过预期值，这些值来自 SciPy 0.13.0 版本
        assert self.gradcalls <= 22, self.gradcalls  # 0.13.0

        # 确保函数的梯度函数调用次数与预期值相符
        # 以下是一些历史数据，用于不同版本的断言，可以根据需要取消注释相应的行来进行验证
        # assert self.gradcalls <= 18, self.gradcalls  # 0.9.0
        # assert self.gradcalls == 18, self.gradcalls  # 0.8.0
        # assert self.gradcalls == 22, self.gradcalls  # 0.7.0

        # 确保函数的迹在给定的范围内，这些数据来自 SciPy 0.7.0 版本
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)
    # 定义一个测试方法，用于测试具有 Hessian 矩阵的 Newton 共轭梯度优化算法
    def test_ncg_hess(self):
        # 如果使用了包装器，则使用 optimize.minimize 函数进行优化
        if self.use_wrapper:
            # 设置优化选项字典，包括最大迭代次数和显示选项
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            # 调用 optimize.minimize 函数进行优化，使用 Newton-CG 方法，
            # 并传入目标函数、起始参数、梯度向量和 Hessian 矩阵等参数
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       hess=self.hess,
                                       args=(), options=opts)['x']
        else:
            # 如果未使用包装器，则使用 optimize.fmin_ncg 函数进行优化
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       fhess=self.hess,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        # 将优化得到的参数赋值给 params
        params = retval

        # 断言优化后的目标函数值与预期解的目标函数值非常接近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 断言函数调用次数不超过特定值，确保函数的稳定性和效率
        assert self.funccalls <= 7, self.funccalls  # gh10673
        assert self.gradcalls <= 18, self.gradcalls  # 0.9.0
        # 下面的注释掉的断言语句是基于不同版本的 SciPy，用于检查特定的函数调用次数
        # assert self.gradcalls == 18, self.gradcalls  # 0.8.0
        # assert self.gradcalls == 22, self.gradcalls  # 0.7.0

        # 断言函数的输出结果与预期结果非常接近，用于验证函数的正确性
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)
    # 测试 Newton-CG 方法中的 Hessian 矩阵乘以向量 p 的计算
    def test_ncg_hessp(self):
        # 如果使用函数包装器，则使用 optimize.minimize 函数进行优化
        if self.use_wrapper:
            # 设置优化选项
            opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
            # 调用 optimize.minimize 进行优化，并返回最优解的参数
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       hessp=self.hessp,
                                       args=(), options=opts)['x']
        else:
            # 使用 optimize.fmin_ncg 函数进行优化
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       fhess_p=self.hessp,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        # 将优化得到的参数保存到 params 中
        params = retval

        # 断言优化后的函数值与已知解的函数值非常接近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # 确保函数调用次数符合预期；这些值是从 SciPy 0.7.0 获得的，不允许增加
        assert self.funccalls <= 7, self.funccalls  # gh10673
        assert self.gradcalls <= 18, self.gradcalls  # 0.9.0

        # 确保函数的计算结果与预期非常接近；这些值是从 SciPy 0.7.0 获得的
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)

    # 测试 COBYQA 方法
    def test_cobyqa(self):
        # 如果使用函数包装器，则使用 optimize.minimize 函数进行优化
        if self.use_wrapper:
            # 调用 optimize.minimize 函数进行优化
            res = optimize.minimize(
                self.func,
                self.startparams,
                method='cobyqa',
                options={'maxiter': self.maxiter, 'disp': self.disp},
            )
            # 断言优化后的函数值与已知解的函数值非常接近
            assert_allclose(res.fun, self.func(self.solution), atol=1e-6)

            # 确保函数调用次数符合预期；这些值是从 SciPy 1.14.0 获得的，不允许增加
            assert self.funccalls <= 45 + 20, self.funccalls
def test_maxfev_test():
    # 使用固定种子生成随机数生成器对象
    rng = np.random.default_rng(271707100830272976862395227613146332411)

    def cost(x):
        # 返回一个随机数乘以1000作为成本函数，模拟永远不收敛的问题
        return rng.random(1) * 1000  # never converged problem

    # 对于不同的最大函数评估次数设置
    for imaxfev in [1, 10, 50]:
        # 对于两种不同的优化方法，测试最大函数评估次数的限制
        for method in ['Powell', 'Nelder-Mead']:
            result = optimize.minimize(cost, rng.random(10),
                                       method=method,
                                       options={'maxfev': imaxfev})
            # 断言函数评估次数与设置的最大函数评估次数相符
            assert result["nfev"] == imaxfev


def test_wrap_scalar_function_with_validation():

    def func_(x):
        # 简单地返回输入值的函数
        return x

    # 获取包装后的函数和调用次数计数器
    fcalls, func = optimize._optimize.\
        _wrap_scalar_function_maxfun_validation(func_, np.asarray(1), 5)

    # 对于一定次数的循环
    for i in range(5):
        func(np.asarray(i))
        # 断言函数调用次数与预期相符
        assert fcalls[0] == i+1

    # 测试函数调用次数超过最大允许次数时是否会引发异常
    msg = "Too many function calls"
    with assert_raises(optimize._optimize._MaxFuncCallError, match=msg):
        func(np.asarray(i))  # exceeded maximum function call

    # 重新获取包装后的函数和调用次数计数器
    fcalls, func = optimize._optimize.\
        _wrap_scalar_function_maxfun_validation(func_, np.asarray(1), 5)

    # 测试当用户提供的目标函数返回非标量值时是否会引发异常
    msg = "The user-provided objective function must return a scalar value."
    with assert_raises(ValueError, match=msg):
        func(np.array([1, 1]))


def test_obj_func_returns_scalar():
    # 测试当用户提供的目标函数返回非标量值时是否会引发异常
    match = ("The user-provided "
             "objective function must "
             "return a scalar value.")
    with assert_raises(ValueError, match=match):
        optimize.minimize(lambda x: x, np.array([1, 1]), method='BFGS')


def test_neldermead_iteration_num():
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    # 使用 Nelder-Mead 方法最小化罗森布罗克函数
    res = optimize._minimize._minimize_neldermead(optimize.rosen, x0,
                                                  xatol=1e-8)
    # 断言迭代次数不超过预期值
    assert res.nit <= 339


def test_neldermead_respect_fp():
    # 测试 Nelder-Mead 方法是否会尊重输入和函数的浮点类型
    x0 = np.array([5.0, 4.0]).astype(np.float32)
    def rosen_(x):
        # 断言输入参数的数据类型为 np.float32
        assert x.dtype == np.float32
        return optimize.rosen(x)

    # 使用 Nelder-Mead 方法最小化罗森布罗克函数
    optimize.minimize(rosen_, x0, method='Nelder-Mead')


def test_neldermead_xatol_fatol():
    # gh4484
    # 测试可以使用指定的 xatol 和 fatol 参数调用函数
    def func(x):
        return x[0] ** 2 + x[1] ** 2

    # 使用 Nelder-Mead 方法最小化函数，指定最大迭代次数、xatol 和 fatol 参数
    optimize._minimize._minimize_neldermead(func, [1, 1], maxiter=2,
                                            xatol=1e-3, fatol=1e-3)


def test_neldermead_adaptive():
    def func(x):
        # 返回输入向量的平方和作为目标函数
        return np.sum(x ** 2)
    p0 = [0.15746215, 0.48087031, 0.44519198, 0.4223638, 0.61505159,
          0.32308456, 0.9692297, 0.4471682, 0.77411992, 0.80441652,
          0.35994957, 0.75487856, 0.99973421, 0.65063887, 0.09626474]

    # 使用 Nelder-Mead 方法最小化目标函数
    res = optimize.minimize(func, p0, method='Nelder-Mead')
    # 断言最优化是否成功
    assert_equal(res.success, False)
    # 使用 optimize 模块中的 minimize 函数来最小化目标函数 func
    # func 是待最小化的目标函数
    # p0 是优化算法的起始点或起始向量
    # method='Nelder-Mead' 指定了使用 Nelder-Mead 方法来进行优化
    # options={'adaptive': True} 是一个选项字典，其中 adaptive=True 表示启用自适应参数调整
    res = optimize.minimize(func, p0, method='Nelder-Mead',
                            options={'adaptive': True})
    
    # 使用断言确保优化成功，即 res.success 应为 True
    assert_equal(res.success, True)
def test_bounded_powell_outsidebounds():
    # 测试有界 Powell 方法，当起始点在边界外部时，最终点应该仍在边界内部
    def func(x):
        return np.sum(x ** 2)
    bounds = (-1, 1), (-1, 1), (-1, 1)
    x0 = [-4, .5, -.8]

    # 我们起始点在边界外部，所以应该会收到警告
    with assert_warns(optimize.OptimizeWarning):
        # 使用 Powell 方法进行优化
        res = optimize.minimize(func, x0, bounds=bounds, method="Powell")
    # 检查结果的 x 是否接近 [0, 0, 0]
    assert_allclose(res.x, np.array([0.] * len(x0)), atol=1e-6)
    # 检查优化是否成功
    assert_equal(res.success, True)
    # 检查优化的状态
    assert_equal(res.status, 0)

    # 然而，如果我们改变 `direc` 参数，使得向量集合不再覆盖参数空间，
    # 那么我们可能不会回到边界内部。这里我们看到第一个参数无法更新！
    direc = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    # 我们起始点在边界外部，所以应该会收到警告
    with assert_warns(optimize.OptimizeWarning):
        # 使用 Powell 方法进行优化，传入 bounds 和 direc 参数
        res = optimize.minimize(func, x0,
                                bounds=bounds, method="Powell",
                                options={'direc': direc})
    # 检查结果的 x 是否接近 [-4, 0, 0]
    assert_allclose(res.x, np.array([-4., 0, 0]), atol=1e-6)
    # 检查优化是否成功
    assert_equal(res.success, False)
    # 检查优化的状态
    assert_equal(res.status, 4)


def test_bounded_powell_vs_powell():
    # 这里我们测试一个例子，有界 Powell 方法和标准 Powell 方法会返回不同的结果。

    # 首先，我们测试一个简单的例子，其中最小值在原点，而边界内的最小值大于原点的最小值。
    def func(x):
        return np.sum(x ** 2)
    bounds = (-5, -1), (-10, -0.1), (1, 9.2), (-4, 7.6), (-15.9, -2)
    x0 = [-2.1, -5.2, 1.9, 0, -2]

    options = {'ftol': 1e-10, 'xtol': 1e-10}

    # 使用 Powell 方法进行优化，无 bounds 参数
    res_powell = optimize.minimize(func, x0, method="Powell", options=options)
    # 检查结果的 x 是否接近 0
    assert_allclose(res_powell.x, 0., atol=1e-6)
    # 检查结果的函数值是否接近 0
    assert_allclose(res_powell.fun, 0., atol=1e-6)

    # 使用有界 Powell 方法进行优化，传入 bounds 参数
    res_bounded_powell = optimize.minimize(func, x0, options=options,
                                           bounds=bounds,
                                           method="Powell")
    # 预期的最优点 p
    p = np.array([-1, -0.1, 1, 0, -2])
    # 检查结果的 x 是否接近 p
    assert_allclose(res_bounded_powell.x, p, atol=1e-6)
    # 检查结果的函数值是否接近 p 的函数值
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-6)

    # 现在我们测试有界 Powell 方法，但 bounds 参数包含 inf 的混合情况。
    bounds = (None, -1), (-np.inf, -.1), (1, np.inf), (-4, None), (-15.9, -2)
    res_bounded_powell = optimize.minimize(func, x0, options=options,
                                           bounds=bounds,
                                           method="Powell")
    # 预期的最优点 p
    p = np.array([-1, -0.1, 1, 0, -2])
    # 检查结果的 x 是否接近 p
    assert_allclose(res_bounded_powell.x, p, atol=1e-6)
    # 检查结果的函数值是否接近 p 的函数值
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-6)

    # 接下来我们测试一个例子，其中全局最小值在边界内部
    # 定义一个用于优化的目标函数 func(x)，这是一个典型的优化问题中的目标函数
    def func(x):
        # 计算目标函数的具体表达式，包含了多个数学运算
        t = np.sin(-x[0]) * np.cos(x[1]) * np.sin(-x[0] * x[1]) * np.cos(x[1])
        t -= np.cos(np.sin(x[1] * x[2]) * np.cos(x[2]))
        # 返回目标函数值的平方
        return t**2

    # 设定变量的边界条件，这里是一个三维变量，每个维度的取值范围为 (-2, 5)
    bounds = [(-2, 5)] * 3
    # 设置初始值，即优化算法的起始点
    x0 = [-0.5, -0.5, -0.5]

    # 使用 Powell 方法进行优化，没有提供边界条件
    res_powell = optimize.minimize(func, x0, method="Powell")
    # 使用 Powell 方法进行优化，提供了边界条件
    res_bounded_powell = optimize.minimize(func, x0,
                                           bounds=bounds,
                                           method="Powell")
    
    # 断言两种 Powell 方法得到的目标函数值接近于预期值
    assert_allclose(res_powell.fun, 0.007136253919761627, atol=1e-6)
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-6)

    # 接下来测试在未提供边界条件 (-inf, inf) 的情况下，与不提供任何边界条件的情况
    bounds = [(-np.inf, np.inf)] * 3

    # 使用 Powell 方法进行优化，提供了 (-inf, inf) 的边界条件
    res_bounded_powell = optimize.minimize(func, x0,
                                           bounds=bounds,
                                           method="Powell")
    # 断言两种 Powell 方法得到的目标函数值应该是相同的
    assert_allclose(res_powell.fun, res_bounded_powell.fun, atol=1e-6)
    # 断言两种 Powell 方法的函数评估次数应该是相同的
    assert_allclose(res_powell.nfev, res_bounded_powell.nfev, atol=1e-6)
    # 断言两种 Powell 方法得到的最优点 x 应该是相同的
    assert_allclose(res_powell.x, res_bounded_powell.x, atol=1e-6)

    # 现在测试当初始点 x0 超出边界时的情况
    x0 = [45.46254415, -26.52351498, 31.74830248]
    bounds = [(-2, 5)] * 3
    # 由于初始点超出了边界，期望会产生一个优化警告
    with assert_warns(optimize.OptimizeWarning):
        res_bounded_powell = optimize.minimize(func, x0,
                                               bounds=bounds,
                                               method="Powell")
    # 断言此时的目标函数值应该接近于预期值
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-6)
def test_onesided_bounded_powell_stability():
    # 当 Powell 方法只在一侧有界时，进行 np.tan 变换以将其转换为完全有界问题。
    # 在这里，我们对单侧有界的 Powell 进行一些简单的测试，其中最优解很大，以测试转换的稳定性。

    # 定义优化方法和约束条件
    kwargs = {'method': 'Powell',
              'bounds': [(-np.inf, 1e6)] * 3,  # 每个变量的边界条件，(-∞, 1e6)
              'options': {'ftol': 1e-8, 'xtol': 1e-8}}  # 优化选项，容差设置

    x0 = [1, 1, 1]  # 初始猜测值

    # 定义函数 f(x) = -np.sum(x)，即目标函数为各变量之和的负值
    def f(x):
        return -np.sum(x)

    # 调用优化函数 minimize，优化目标函数 f(x)，传入初始猜测值和参数 kwargs
    res = optimize.minimize(f, x0, **kwargs)

    # 断言优化结果的目标函数值与预期值的接近程度在指定的绝对容差范围内
    assert_allclose(res.fun, -3e6, atol=1e-4)

    # 定义函数 f(x)，其中 df/dx 逐渐变小
    def f(x):
        return -np.abs(np.sum(x)) ** (0.1) * (1 if np.all(x > 0) else -1)

    # 再次调用优化函数 minimize，传入新定义的目标函数 f(x) 和之前的参数 kwargs
    res = optimize.minimize(f, x0, **kwargs)

    # 断言优化结果的目标函数值与预期值的接近程度在默认的相对容差范围内
    assert_allclose(res.fun, -(3e6) ** (0.1))

    # 定义函数 f(x)，其中 df/dx 逐渐变大
    def f(x):
        return -np.abs(np.sum(x)) ** 10 * (1 if np.all(x > 0) else -1)

    # 再次调用优化函数 minimize，传入新定义的目标函数 f(x) 和之前的参数 kwargs
    res = optimize.minimize(f, x0, **kwargs)

    # 断言优化结果的目标函数值与预期值的接近程度在指定的相对容差范围内
    assert_allclose(res.fun, -(3e6) ** 10, rtol=1e-7)

    # 定义函数 f(x)，其中某些变量 df/dx 变大，另一些变小
    def f(x):
        t = -np.abs(np.sum(x[:2])) ** 5 - np.abs(np.sum(x[2:])) ** (0.1)
        t *= (1 if np.all(x > 0) else -1)
        return t

    kwargs['bounds'] = [(-np.inf, 1e3)] * 3  # 修改边界条件为每个变量 (-∞, 1e3)
    # 再次调用优化函数 minimize，传入新定义的目标函数 f(x) 和更新后的参数 kwargs
    res = optimize.minimize(f, x0, **kwargs)

    # 断言优化结果的目标函数值与预期值的接近程度在指定的相对容差范围内
    assert_allclose(res.fun, -(2e3) ** 5 - (1e6) ** (0.1), rtol=1e-7)
    def test_bfgs_nan_return(self):
        # 测试当函数返回 NaN 时的边界情况。参见 gh-4793.

        # 第一种情况：第一次调用返回 NaN。
        def func(x):
            return np.nan
        # 忽略无效数错误状态
        with np.errstate(invalid='ignore'):
            # 使用 BFGS 方法最小化函数 func，起始点为 0
            result = optimize.minimize(func, 0)

        # 断言最小化结果的函数值为 NaN
        assert np.isnan(result['fun'])
        # 断言最小化未成功
        assert result['success'] is False

        # 第二种情况：第二次调用返回 NaN。
        def func(x):
            return 0 if x == 0 else np.nan
        # 定义 func 的导数函数 fprime
        def fprime(x):
            return np.ones_like(x)  # 避开零点
        # 忽略无效数错误状态
        with np.errstate(invalid='ignore'):
            # 使用 BFGS 方法最小化函数 func，起始点为 0，指定数值梯度函数为 fprime
            result = optimize.minimize(func, 0, jac=fprime)

        # 断言最小化结果的函数值为 NaN
        assert np.isnan(result['fun'])
        # 断言最小化未成功
        assert result['success'] is False

    def test_bfgs_numerical_jacobian(self):
        # 使用数值雅可比矩阵的 BFGS 方法，并使用随机向量定义 epsilon 参数。
        epsilon = np.sqrt(np.spacing(1.)) * np.random.rand(len(self.solution))

        # 使用 fmin_bfgs 函数最小化 self.func 函数，起始点为 self.startparams，
        # epsilon 参数为上述定义，args 为空元组，最大迭代次数为 self.maxiter，不显示详细信息。
        params = optimize.fmin_bfgs(self.func, self.startparams,
                                    epsilon=epsilon, args=(),
                                    maxiter=self.maxiter, disp=False)

        # 断言最小化后的 self.func(params) 与 self.func(self.solution) 的值在给定的公差范围内接近
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_finite_differences_jac(self):
        # 使用有限差分法计算雅可比矩阵的各种方法（'BFGS', 'CG', 'TNC'）和雅可比矩阵类型（'2-point', '3-point', None）的组合。
        methods = ['BFGS', 'CG', 'TNC']
        jacs = ['2-point', '3-point', None]
        # 遍历方法和雅可比矩阵类型的所有组合
        for method, jac in itertools.product(methods, jacs):
            # 使用指定方法和雅可比矩阵类型最小化 self.func 函数，起始点为 self.startparams
            result = optimize.minimize(self.func, self.startparams,
                                       method=method, jac=jac)
            # 断言最小化后的 self.func(result.x) 与 self.func(self.solution) 的值在给定的公差范围内接近
            assert_allclose(self.func(result.x), self.func(self.solution),
                            atol=1e-6)
    def test_finite_differences_hess(self):
        """
        # 测试需要 Hessian 的所有方法是否可以使用有限差分
        # 对于 Newton-CG、trust-ncg、trust-krylov 方法，使用有限差分估计的 Hessian 被包装在 hessp 函数中
        # dogleg、trust-exact 实际上需要真实的 Hessian，所以它们被排除在外
        """
        methods = ['trust-constr', 'Newton-CG', 'trust-ncg', 'trust-krylov']
        hesses = FD_METHODS + (optimize.BFGS,)
        for method, hess in itertools.product(methods, hesses):
            if hess is optimize.BFGS:
                hess = hess()
            result = optimize.minimize(self.func, self.startparams,
                                       method=method, jac=self.grad,
                                       hess=hess)
            assert result.success

        """
        # 检查这些方法是否要求某种形式的 Hessian 指定
        # Newton-CG 会创建自己的 hessp，trust-constr 也不需要指定 hess
        """
        methods = ['trust-ncg', 'trust-krylov', 'dogleg', 'trust-exact']
        for method in methods:
            with pytest.raises(ValueError):
                optimize.minimize(self.func, self.startparams,
                                  method=method, jac=self.grad,
                                  hess=None)

    def test_bfgs_gh_2169(self):
        """
        # 检查 BFGS 是否避免在同一点连续两次评估
        """
        def f(x):
            if x < 0:
                return 1.79769313e+308
            else:
                return x + 1./x
        xs = optimize.fmin_bfgs(f, [10.], disp=False)
        assert_allclose(xs, 1.0, rtol=1e-4, atol=1e-4)

    def test_bfgs_double_evaluations(self):
        """
        # 检查 BFGS 是否不会连续两次在同一点进行评估
        """
        def f(x):
            xp = x[0]
            assert xp not in seen
            seen.add(xp)
            return 10*x**2, 20*x

        seen = set()
        optimize.minimize(f, -100, method='bfgs', jac=True, tol=1e-7)

    def test_l_bfgs_b(self):
        """
        # 有界约束的有限记忆 BFGS 算法
        """
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams,
                                        self.grad, args=(),
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        """
        # 确保函数调用计数是 'known good'；这些来自于 SciPy 0.7.0。不允许它们增加。
        """
        assert self.funccalls == 7, self.funccalls
        assert self.gradcalls == 5, self.gradcalls

        """
        # 确保函数行为相同；这来自于 SciPy 0.7.0 中在 gh10673 中修复的测试
        """
        assert_allclose(self.trace[3:5],
                        [[8.117083e-16, -5.196198e-01, 4.897617e-01],
                         [0., -0.52489628, 0.48753042]],
                        atol=1e-14, rtol=1e-7)
    def test_l_bfgs_b_numjac(self):
        # L-BFGS-B算法使用数值Jacobi矩阵
        # 使用optimize.fmin_l_bfgs_b函数进行优化，传入目标函数self.func、初始参数self.startparams，
        # 启用近似梯度approx_grad=True，设定最大迭代次数maxiter=self.maxiter
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams,
                                        approx_grad=True,
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval  # 解析优化结果为params、fopt、d

        # 断言优化后的目标函数值与预期解self.solution对应的目标函数值相近，容差为1e-6
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_l_bfgs_b_funjac(self):
        # L-BFGS-B算法使用结合的目标函数和Jacobi矩阵
        # 定义组合函数fun(x)，返回目标函数值self.func(x)和梯度self.grad(x)
        def fun(x):
            return self.func(x), self.grad(x)

        # 使用optimize.fmin_l_bfgs_b函数进行优化，传入fun函数作为目标函数、初始参数self.startparams，
        # 设定最大迭代次数maxiter=self.maxiter
        retval = optimize.fmin_l_bfgs_b(fun, self.startparams,
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval  # 解析优化结果为params、fopt、d

        # 断言优化后的目标函数值与预期解self.solution对应的目标函数值相近，容差为1e-6
        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_l_bfgs_b_maxiter(self):
        # gh7854
        # 确保不超过设定的最大迭代次数
        # 定义一个Callback类，用于优化过程中的回调操作
        class Callback:
            def __init__(self):
                self.nit = 0  # 初始化迭代次数为0
                self.fun = None  # 初始化目标函数值为None
                self.x = None  # 初始化当前参数值为None

            def __call__(self, x):
                self.x = x  # 更新当前参数值
                self.fun = optimize.rosen(x)  # 计算当前参数值对应的Rosenbrock函数值
                self.nit += 1  # 迭代次数加1

        c = Callback()  # 创建Callback对象c
        # 使用optimize.minimize函数进行优化，传入Rosenbrock函数optimize.rosen、初始参数[0., 0.]，
        # 方法为'l-bfgs-b'，设定回调函数callback=c、选项设定最大迭代次数为5
        res = optimize.minimize(optimize.rosen, [0., 0.], method='l-bfgs-b',
                                callback=c, options={'maxiter': 5})

        # 断言优化结果的迭代次数res.nit等于设定的最大迭代次数5
        assert_equal(res.nit, 5)
        # 断言优化结果的参数值res.x与回调中记录的参数值c.x相近
        assert_almost_equal(res.x, c.x)
        # 断言优化结果的目标函数值res.fun与回调中记录的目标函数值c.fun相近
        assert_almost_equal(res.fun, c.fun)
        # 断言优化结果的状态为1（未知状态，因迭代次数到达上限而终止）
        assert_equal(res.status, 1)
        # 断言优化未成功
        assert res.success is False
        # 断言优化消息为迭代次数到达上限的提示信息
        assert_equal(res.message,
                     'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT')

    def test_minimize_l_bfgs_b(self):
        # 使用L-BFGS-B方法进行最小化
        opts = {'disp': False, 'maxiter': self.maxiter}  # 设置选项，禁止显示和设定最大迭代次数
        # 使用optimize.minimize函数进行优化，传入目标函数self.func、初始参数self.startparams，
        # 方法为'L-BFGS-B'，梯度函数为self.grad，选项为opts
        r = optimize.minimize(self.func, self.startparams,
                              method='L-BFGS-B', jac=self.grad,
                              options=opts)
        # 断言优化后的目标函数值与预期解self.solution对应的目标函数值相近，容差为1e-6
        assert_allclose(self.func(r.x), self.func(self.solution),
                        atol=1e-6)
        # 断言梯度计算次数self.gradcalls与优化结果的梯度评估次数r.njev相等
        assert self.gradcalls == r.njev

        self.funccalls = self.gradcalls = 0  # 重置函数调用计数
        # 使用近似Jacobi矩阵进行优化
        ra = optimize.minimize(self.func, self.startparams,
                               method='L-BFGS-B', options=opts)
        # 断言近似Jacobi矩阵中的函数评估次数self.funccalls大于普通Jacobi矩阵的评估次数r.nfev
        # assert_(ra.nfev > r.nfev)  # 这行注释掉的断言在代码中未被使用
        assert self.funccalls == ra.nfev  # 断言函数调用计数self.funccalls与优化结果的函数评估次数ra.nfev相等
        # 断言优化后的目标函数值与预期解self.solution对应的目标函数值相近，容差为1e-6
        assert_allclose(self.func(ra.x), self.func(self.solution),
                        atol=1e-6)

        self.funccalls = self.gradcalls = 0  # 重置函数调用计数
        # 使用3点近似Jacobi矩阵进行优化
        ra = optimize.minimize(self.func, self.startparams, jac='3-point',
                               method='L-BFGS-B', options=opts)
        # 断言函数调用计数self.funccalls与优化结果的函数评估次数ra.nfev相等
        assert self.funccalls == ra.nfev
        # 断言优化后的目标函数值与预期解self.solution对应的目标函数值相近，容差为1e-6
        assert_allclose(self.func(ra.x), self.func(self.solution),
                        atol=1e-6)
    def test_minimize_l_bfgs_b_ftol(self):
        # Check that the `ftol` parameter in L-BFGS-B optimization works as expected

        # Initialize v0 to None
        v0 = None
        
        # Iterate over different tolerance values
        for tol in [1e-1, 1e-4, 1e-7, 1e-10]:
            # Define optimization options dictionary
            opts = {'disp': False, 'maxiter': self.maxiter, 'ftol': tol}
            
            # Perform optimization using L-BFGS-B method
            sol = optimize.minimize(self.func, self.startparams,
                                    method='L-BFGS-B', jac=self.grad,
                                    options=opts)
            
            # Compute the function value at the solution
            v = self.func(sol.x)

            # Compare current function value with initial value v0
            if v0 is None:
                v0 = v
            else:
                assert v < v0

            # Assert that the current function value is close to the known solution
            assert_allclose(v, self.func(self.solution), rtol=tol)

    def test_minimize_l_bfgs_maxls(self):
        # Check that the `maxls` parameter is passed correctly to the L-BFGS-B routine
        
        # Perform optimization using L-BFGS-B method with a specified maxls value
        sol = optimize.minimize(optimize.rosen, np.array([-1.2, 1.0]),
                                method='L-BFGS-B', jac=optimize.rosen_der,
                                options={'disp': False, 'maxls': 1})
        
        # Assert that the optimization did not succeed
        assert not sol.success

    def test_minimize_l_bfgs_b_maxfun_interruption(self):
        # Test case for issue gh-6162
        
        # Define the Rosenbrock function and its derivative
        f = optimize.rosen
        g = optimize.rosen_der
        
        # List to store function values during optimization
        values = []
        
        # Initial guess for the optimization
        x0 = np.full(7, 1000)

        # Define objective function to be minimized
        def objfun(x):
            # Compute function value and append it to values list
            value = f(x)
            values.append(value)
            return value

        # Find a good stopping point for maxfun between 100 and 300 evaluations
        low, medium, high = 30, 100, 300
        
        # Perform optimization with a specified maxfun
        optimize.fmin_l_bfgs_b(objfun, x0, fprime=g, maxfun=high)
        
        # Find the maximum function value and its index in the values list
        v, k = max((y, i) for i, y in enumerate(values[medium:]))
        
        # Determine the actual maxfun based on the index found
        maxfun = medium + k
        
        # Define target value as the minimum of the first 30 function evaluations
        target = min(values[:low])
        
        # Perform another optimization with a specified maxfun and compare results
        xmin, fmin, d = optimize.fmin_l_bfgs_b(f, x0, fprime=g, maxfun=maxfun)
        
        # Assert that the final minimized function value is less than the target value
        assert_array_less(fmin, target)
    def test_gh10771(self):
        # check that minimize passes bounds and constraints to a custom
        # minimizer without altering them.
        bounds = [(-2, 2), (0, 3)]  # 定义变量范围
        constraints = 'constraints'  # 定义约束条件

        def custmin(fun, x0, **options):
            assert options['bounds'] is bounds  # 断言确保传递的变量范围未被修改
            assert options['constraints'] is constraints  # 断言确保传递的约束条件未被修改
            return optimize.OptimizeResult()  # 返回优化结果对象

        x0 = [1, 1]  # 初始变量值
        optimize.minimize(optimize.rosen, x0, method=custmin,
                          bounds=bounds, constraints=constraints)
    # 定义一个测试函数，用于验证 minimize() 函数中的 tol 参数是否起作用
    def test_minimize_tol_parameter(self):
        # 定义一个测试函数 func(z)，计算 z[0]**2 * z[1]**2 + z[0]**4 + 1 的值并返回
        def func(z):
            x, y = z
            return x**2 * y**2 + x**4 + 1
        
        # 定义一个计算梯度的函数 dfunc(z)，返回数组 [2*x*y**2 + 4*x**3, 2*x**2*y]
        def dfunc(z):
            x, y = z
            return np.array([2*x*y**2 + 4*x**3, 2*x**2*y])
        
        # 遍历不同的优化方法
        for method in ['nelder-mead', 'powell', 'cg', 'bfgs',
                       'newton-cg', 'l-bfgs-b', 'tnc',
                       'cobyla', 'cobyqa', 'slsqp']:
            # 根据方法选择是否计算雅可比矩阵
            if method in ('nelder-mead', 'powell', 'cobyla', 'cobyqa'):
                jac = None  # 对于部分方法，不需要提供雅可比矩阵
            else:
                jac = dfunc  # 对于其他方法，使用预定义的梯度函数 dfunc
            
            # 调用 optimize.minimize() 函数，使用给定的方法和不同的 tol 参数值进行优化
            sol1 = optimize.minimize(func, [2, 2], jac=jac, tol=1e-10,
                                     method=method)
            sol2 = optimize.minimize(func, [2, 2], jac=jac, tol=1.0,
                                     method=method)
            
            # 断言，确保在不同 tol 参数下，sol1 的函数值小于 sol2 的函数值
            assert func(sol1.x) < func(sol2.x), \
                   f"{method}: {func(sol1.x)} vs. {func(sol2.x)}"

    # 使用 pytest 的装饰器标记此测试为 "fail_slow"，最多允许 10 次失败
    @pytest.mark.fail_slow(10)
    # 使用 pytest 的装饰器过滤忽略特定的用户警告
    @pytest.mark.filterwarnings('ignore::UserWarning')
    # 使用 pytest 的装饰器过滤忽略特定的运行时警告，并且注释说明该过滤是因为 gh-18547
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # See gh-18547
    # 使用 pytest 的装饰器参数化 method 参数，传入多个优化方法名称以及预定义的 MINIMIZE_METHODS 列表
    @pytest.mark.parametrize('method',
                             ['fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs',
                              'fmin_ncg', 'fmin_l_bfgs_b', 'fmin_tnc',
                              'fmin_slsqp'] + MINIMIZE_METHODS)
    def test_minimize_callback_copies_array(self, method):
        # Check that arrays passed to callbacks are not modified
        # inplace by the optimizer afterward

        # 根据给定的优化方法选择相应的函数或者函数组合
        if method in ('fmin_tnc', 'fmin_l_bfgs_b'):
            def func(x):
                return optimize.rosen(x), optimize.rosen_der(x)
        else:
            func = optimize.rosen
            jac = optimize.rosen_der
            hess = optimize.rosen_hess

        # 初始化初始数组
        x0 = np.zeros(10)

        # 设置选项参数
        kwargs = {}
        if method.startswith('fmin'):
            routine = getattr(optimize, method)
            if method == 'fmin_slsqp':
                kwargs['iter'] = 5
            elif method == 'fmin_tnc':
                kwargs['maxfun'] = 100
            elif method in ('fmin', 'fmin_powell'):
                kwargs['maxiter'] = 3500
            else:
                kwargs['maxiter'] = 5
        else:
            def routine(*a, **kw):
                kw['method'] = method
                return optimize.minimize(*a, **kw)

            if method == 'tnc':
                kwargs['options'] = dict(maxfun=100)
            else:
                kwargs['options'] = dict(maxiter=5)

        # 根据方法选择特定的参数设置
        if method in ('fmin_ncg',):
            kwargs['fprime'] = jac
        elif method in ('newton-cg',):
            kwargs['jac'] = jac
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = jac
            kwargs['hess'] = hess

        # 运行带有回调函数的优化程序
        results = []

        # 定义回调函数，确保传递给回调函数的数组没有被优化器原地修改
        def callback(x, *args, **kwargs):
            assert not isinstance(x, optimize.OptimizeResult)
            results.append((x, np.copy(x)))

        # 调用相应的优化函数进行优化，并传入回调函数及其它参数
        routine(func, x0, callback=callback, **kwargs)

        # 检查返回的数组与它们的复本是否一致，并且没有内存重叠
        assert len(results) > 2
        assert all(np.all(x == y) for x, y in results)
        combinations = itertools.combinations(results, 2)
        assert not any(np.may_share_memory(x[0], y[0]) for x, y in combinations)
    def test_no_increase(self, method):
        # 检查求解器不会返回比初始点更差的值。

        def func(x):
            return (x - 1)**2
        # 定义简单的二次函数

        def bad_grad(x):
            # 故意设置无效的梯度函数，模拟线搜索失败的情况
            return 2*(x - 1) * (-1) - 2
        # 定义一个有意设计为无效的梯度函数

        x0 = np.array([2.0])
        # 设置初始点

        f0 = func(x0)
        # 计算初始点的函数值

        jac = bad_grad
        # 将梯度函数设为bad_grad

        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)
        # 根据方法选择选项，最大函数调用次数或最大迭代次数为20

        if method in ['nelder-mead', 'powell', 'cobyla', 'cobyqa']:
            jac = None
        # 对于特定方法，不使用梯度

        sol = optimize.minimize(func, x0, jac=jac, method=method,
                                options=options)
        # 使用给定方法和选项进行优化求解

        assert_equal(func(sol.x), sol.fun)
        # 断言优化后的函数值与目标函数值相等

        if method == 'slsqp':
            pytest.xfail("SLSQP returns slightly worse")
            # 如果是SLSQP方法，预期失败，因为SLSQP可能会返回稍差的结果
        assert func(sol.x) <= f0
        # 断言优化后的函数值不大于初始点的函数值

    def test_slsqp_respect_bounds(self):
        # 对gh-3108进行回归测试

        def f(x):
            return sum((x - np.array([1., 2., 3., 4.]))**2)
        # 定义一个目标函数

        def cons(x):
            a = np.array([[-1, -1, -1, -1], [-3, -3, -2, -1]])
            return np.concatenate([np.dot(a, x) + np.array([5, 10]), x])
        # 定义约束条件

        x0 = np.array([0.5, 1., 1.5, 2.])
        # 设置初始点

        res = optimize.minimize(f, x0, method='slsqp',
                                constraints={'type': 'ineq', 'fun': cons})
        # 使用SLSQP方法进行优化，添加约束条件

        assert_allclose(res.x, np.array([0., 2, 5, 8])/3, atol=1e-12)
        # 断言优化结果与预期结果在指定精度下相等

    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                                        'Newton-CG', 'L-BFGS-B', 'SLSQP',
                                        'trust-constr', 'dogleg', 'trust-ncg',
                                        'trust-exact', 'trust-krylov',
                                        'cobyqa'])
    def test_respect_maxiter(self, method):
        # 检查迭代次数是否等于max_iter，假设在收敛之前未能达到

        MAXITER = 4
        # 设置最大迭代次数

        x0 = np.zeros(10)
        # 初始化点为10维零向量

        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der,
                            optimize.rosen_hess, None, None)
        # 创建标量函数对象，使用Rosenbrock函数的导数和Hessian矩阵

        # 设置选项
        kwargs = {'method': method, 'options': dict(maxiter=MAXITER)}

        if method in ('Newton-CG',):
            kwargs['jac'] = sf.grad
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = sf.grad
            kwargs['hess'] = sf.hess
        # 对于特定方法，设置梯度和Hessian矩阵的选项

        sol = optimize.minimize(sf.fun, x0, **kwargs)
        # 使用给定方法和选项进行优化求解

        assert sol.nit == MAXITER
        # 断言迭代次数与设定的最大迭代次数相等
        assert sol.nfev >= sf.nfev
        # 断言函数评估次数不少于标量函数对象的函数评估次数
        if hasattr(sol, 'njev'):
            assert sol.njev >= sf.ngev
        # 如果有njev属性，断言雅可比矩阵评估次数不少于标量函数对象的梯度评估次数

        # 方法特定的测试
        if method == 'SLSQP':
            assert sol.status == 9  # Iteration limit reached
            # 断言状态为9，表示达到迭代限制
        elif method == 'cobyqa':
            assert sol.status == 6  # Iteration limit reached
            # 断言状态为6，表示达到迭代限制
    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell',
                                        'fmin', 'fmin_powell'])
    # 使用 pytest 的参数化装饰器，指定不同的方法名称作为测试用例的参数
    def test_runtime_warning(self, method):
        # 初始化一个长度为 10 的全零数组作为优化函数的初始点
        x0 = np.zeros(10)
        # 创建 ScalarFunction 对象，用于优化 Rosenbrock 函数
        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der,
                            optimize.rosen_hess, None, None)
        # 设定优化选项，最大迭代次数为 1，显示优化过程
        options = {"maxiter": 1, "disp": True}
        # 使用 pytest 的 warns 上下文管理器捕获 RuntimeWarning 异常
        with pytest.warns(RuntimeWarning,
                          match=r'Maximum number of iterations'):
            # 根据不同的方法名调用不同的优化函数
            if method.startswith('fmin'):
                routine = getattr(optimize, method)
                routine(sf.fun, x0, **options)
            else:
                optimize.minimize(sf.fun, x0, method=method, options=options)

    def test_respect_maxiter_trust_constr_ineq_constraints(self):
        # 设定最大迭代次数为 4
        MAXITER = 4
        # 定义 Rosenbrock 函数及其导数和黑塞矩阵
        f = optimize.rosen
        jac = optimize.rosen_der
        hess = optimize.rosen_hess

        # 定义一个约束函数，用于不等式约束
        def fun(x):
            return np.array([0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])

        # 定义不等式约束
        cons = ({'type': 'ineq',
                 'fun': fun},)

        # 初始化一个长度为 10 的全零数组作为优化函数的初始点
        x0 = np.zeros(10)
        # 使用 trust-constr 方法进行优化，设定最大迭代次数为 MAXITER
        sol = optimize.minimize(f, x0, constraints=cons, jac=jac, hess=hess,
                                method='trust-constr',
                                options=dict(maxiter=MAXITER))
        # 断言优化迭代次数与设定的最大迭代次数相等
        assert sol.nit == MAXITER

    def test_minimize_automethod(self):
        # 定义一个简单的二次函数
        def f(x):
            return x**2

        # 定义一个线性约束函数
        def cons(x):
            return x - 2

        # 初始化一个初始点为 10 的数组
        x0 = np.array([10.])
        # 分别使用不同的方法进行优化，并且断言优化成功
        sol_0 = optimize.minimize(f, x0)
        sol_1 = optimize.minimize(f, x0, constraints=[{'type': 'ineq',
                                                       'fun': cons}])
        sol_2 = optimize.minimize(f, x0, bounds=[(5, 10)])
        sol_3 = optimize.minimize(f, x0,
                                  constraints=[{'type': 'ineq', 'fun': cons}],
                                  bounds=[(5, 10)])
        sol_4 = optimize.minimize(f, x0,
                                  constraints=[{'type': 'ineq', 'fun': cons}],
                                  bounds=[(1, 10)])
        # 断言所有优化结果均成功
        for sol in [sol_0, sol_1, sol_2, sol_3, sol_4]:
            assert sol.success
        # 使用 assert_allclose 断言优化结果与预期值在指定精度范围内相等
        assert_allclose(sol_0.x, 0, atol=1e-7)
        assert_allclose(sol_1.x, 2, atol=1e-7)
        assert_allclose(sol_2.x, 5, atol=1e-7)
        assert_allclose(sol_3.x, 5, atol=1e-7)
        assert_allclose(sol_4.x, 2, atol=1e-7)

    def test_minimize_coerce_args_param(self):
        # 用于回归测试 gh-3503 的函数 Y
        def Y(x, c):
            return np.sum((x-c)**2)

        # Y 对 x 的导数
        def dY_dx(x, c=None):
            return 2*(x-c)

        # 定义一个数组 c 作为参数
        c = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        # 随机初始化一个长度与 c 相同的数组作为初始点
        xinit = np.random.randn(len(c))
        # 使用 BFGS 方法进行优化，传入 Y 函数及其导数
        optimize.minimize(Y, xinit, jac=dY_dx, args=(c), method="BFGS")
    def test_initial_step_scaling(self):
        # 检查优化器初始步长是否合理，即使函数和梯度很大

        scales = [1e-50, 1, 1e50]
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']

        def f(x):
            # 如果第一个步长为空并且 x[0] 不等于 x0[0]，设置第一个步长为 x[0] 与 x0[0] 的绝对差值
            if first_step_size[0] is None and x[0] != x0[0]:
                first_step_size[0] = abs(x[0] - x0[0])
            # 如果 x 的绝对值最大超过 1e4，抛出错误
            if abs(x).max() > 1e4:
                raise AssertionError("Optimization stepped far away!")
            # 返回 scale*(x[0] - 1)**2 的值作为函数值
            return scale*(x[0] - 1)**2

        def g(x):
            # 返回一个包含梯度值的 NumPy 数组，梯度为 scale*(x[0] - 1)
            return np.array([scale*(x[0] - 1)])

        # 遍历 scales 和 methods 的组合
        for scale, method in itertools.product(scales, methods):
            if method in ('CG', 'BFGS'):
                options = dict(gtol=scale*1e-8)
            else:
                options = dict()

            if scale < 1e-10 and method in ('L-BFGS-B', 'Newton-CG'):
                # 如果 scale 小于 1e-10 并且方法为 'L-BFGS-B' 或 'Newton-CG'，跳过当前循环
                # XXX: 如果遇到小梯度，则返回初始点
                continue

            x0 = [-1.0]
            first_step_size = [None]
            # 使用 optimize.minimize 函数进行优化
            res = optimize.minimize(f, x0, jac=g, method=method,
                                    options=options)

            err_msg = f"{method} {scale}: {first_step_size}: {res}"

            # 断言优化成功
            assert res.success, err_msg
            # 断言优化结果的最优点接近 [1.0]
            assert_allclose(res.x, [1.0], err_msg=err_msg)
            # 断言迭代次数小于等于 3
            assert res.nit <= 3, err_msg

            if scale > 1e-10:
                if method in ('CG', 'BFGS'):
                    # 如果方法是 'CG' 或 'BFGS'，断言第一个步长接近 1.01
                    assert_allclose(first_step_size[0], 1.01, err_msg=err_msg)
                else:
                    # 对于 'Newton-CG' 和 'L-BFGS-B'，第一个步长的逻辑不同，
                    # 但都应该在 0.5 和 3 之间
                    assert first_step_size[0] > 0.5 and first_step_size[0] < 3, err_msg
            else:
                # 步长的上界为 ||grad||，因此进行线搜索会产生许多小步长
                pass

    @pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg', 'bfgs',
                                        'newton-cg', 'l-bfgs-b', 'tnc',
                                        'cobyla', 'cobyqa', 'slsqp',
                                        'trust-constr', 'dogleg', 'trust-ncg',
                                        'trust-exact', 'trust-krylov'])
    def test_nan_values(self, method):
        # 检查 NaN 值导致的失败退出状态
        np.random.seed(1234)

        count = [0]  # 初始化一个计数器列表，用于记录函数调用次数

        def func(x):
            return np.nan  # 返回 NaN

        def func2(x):
            count[0] += 1  # 每次调用增加计数器值
            if count[0] > 2:
                return np.nan  # 超过两次调用返回 NaN
            else:
                return np.random.rand()  # 返回随机数

        def grad(x):
            return np.array([1.0])  # 返回包含一个浮点数 1.0 的数组

        def hess(x):
            return np.array([[1.0]])  # 返回一个包含单个元素 1.0 的二维数组

        x0 = np.array([1.0])  # 创建包含单个浮点数 1.0 的 NumPy 数组作为初始点

        # 根据方法确定是否需要梯度和黑塞矩阵
        needs_grad = method in ('newton-cg', 'trust-krylov', 'trust-exact',
                                'trust-ncg', 'dogleg')
        needs_hess = method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                                'dogleg')

        # 函数和对应的梯度、黑塞矩阵的列表
        funcs = [func, func2]
        grads = [grad] if needs_grad else [grad, None]
        hesss = [hess] if needs_hess else [hess, None]

        # 根据方法选择不同的优化选项
        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)

        # 忽略无效值错误和特定运行时警告
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.*")
            sup.filter(RuntimeWarning, ".*does not use Hessian.*")
            sup.filter(RuntimeWarning, ".*does not use gradient.*")

            # 使用 product 函数生成函数、梯度、黑塞矩阵的所有组合，并进行优化
            for f, g, h in itertools.product(funcs, grads, hesss):
                count = [0]  # 每次迭代重新初始化计数器
                sol = optimize.minimize(f, x0, jac=g, hess=h, method=method,
                                        options=options)
                assert_equal(sol.success, False)  # 断言优化失败

    @pytest.mark.parametrize('method', ['nelder-mead', 'cg', 'bfgs',
                                        'l-bfgs-b', 'tnc',
                                        'cobyla', 'cobyqa', 'slsqp',
                                        'trust-constr', 'dogleg', 'trust-ncg',
                                        'trust-exact', 'trust-krylov'])
    def test_duplicate_evaluations(self, method):
        # 检查各种方法中是否存在重复评估
        jac = hess = None
        if method in ('newton-cg', 'trust-krylov', 'trust-exact',
                      'trust-ncg', 'dogleg'):
            jac = self.grad  # 如果方法需要梯度，则使用预定义的梯度函数

        if method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                      'dogleg'):
            hess = self.hess  # 如果方法需要黑塞矩阵，则使用预定义的黑塞矩阵函数

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            # 忽略特定运行时警告
            sup.filter(UserWarning, "delta_grad == 0.*")
            # 调用 optimize.minimize 进行优化，忽略无效值警告
            optimize.minimize(self.func, self.startparams,
                              method=method, jac=jac, hess=hess)

        # 检查优化过程中是否存在重复评估
        for i in range(1, len(self.trace)):
            if np.array_equal(self.trace[i - 1], self.trace[i]):
                raise RuntimeError(
                    f"Duplicate evaluations made by {method}")

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('method', MINIMIZE_METHODS_NEW_CB)
    @pytest.mark.parametrize('new_cb_interface', [0, 1, 2])
    def test_callback_stopiteration(self, method, new_cb_interface):
        # 检查如果回调函数引发 StopIteration，优化是否以与限制迭代相同的结果终止

        def f(x):
            f.flag = False  # 检查在 StopIteration 后 f 函数是否被调用
            return optimize.rosen(x)
        f.flag = False

        def g(x):
            f.flag = False
            return optimize.rosen_der(x)

        def h(x):
            f.flag = False
            return optimize.rosen_hess(x)

        maxiter = 5

        # 根据 new_cb_interface 的不同值选择不同的回调接口函数
        if new_cb_interface == 1:
            def callback_interface(*, intermediate_result):
                assert intermediate_result.fun == f(intermediate_result.x)
                callback()
        elif new_cb_interface == 2:
            class Callback:
                def __call__(self, intermediate_result: OptimizeResult):
                    assert intermediate_result.fun == f(intermediate_result.x)
                    callback()
            callback_interface = Callback()
        else:
            def callback_interface(xk, *args):  # type: ignore[misc]
                callback()

        # 定义回调函数，用于在达到最大迭代次数时触发 StopIteration
        def callback():
            callback.i += 1
            callback.flag = False
            if callback.i == maxiter:
                callback.flag = True
                raise StopIteration()
        callback.i = 0
        callback.flag = False

        kwargs = {'x0': [1.1]*5, 'method': method,
                  'fun': f, 'jac': g, 'hess': h}

        # 执行优化过程，传入回调函数作为 callback_interface
        res = optimize.minimize(**kwargs, callback=callback_interface)

        # 对特定方法进行额外的最大迭代次数校验
        if method == 'nelder-mead':
            maxiter = maxiter + 1  # nelder-mead 方法计数方式不同
        if method == 'cobyqa':
            ref = optimize.minimize(**kwargs, options={'maxfev': maxiter})
            assert res.nfev == ref.nfev == maxiter
        else:
            ref = optimize.minimize(**kwargs, options={'maxiter': maxiter})
            assert res.nit == ref.nit == maxiter

        # 校验优化结果
        assert res.fun == ref.fun
        assert_equal(res.x, ref.x)
        assert res.status == (3 if method in [
            'trust-constr',
            'cobyqa',
        ] else 99)

    def test_ndim_error(self):
        # 测试当 'x0' 的维度超过一维时是否引发 ValueError 异常
        msg = "'x0' must only have one dimension."
        with assert_raises(ValueError, match=msg):
            optimize.minimize(lambda x: x, np.ones((2, 1)))

    @pytest.mark.parametrize('method', ('nelder-mead', 'l-bfgs-b', 'tnc',
                                        'powell', 'cobyla', 'cobyqa',
                                        'trust-constr'))


这些注释详细解释了每行代码的作用，确保了代码的每个部分都得到了适当的说明。
    # 定义一个测试函数，用于测试在指定方法下的 optimize.minimize 函数的边界条件处理
    def test_minimize_invalid_bounds(self, method):
        # 定义一个简单的目标函数，计算输入向量 x 各元素平方和
        def f(x):
            return np.sum(x**2)

        # 创建 Bounds 对象，指定变量 x 的上下界
        bounds = Bounds([1, 2], [3, 4])
        # 设置错误消息，用于检查是否抛出 ValueError 异常
        msg = 'The number of bounds is not compatible with the length of `x0`.'
        # 使用 pytest 的上下文管理器检查是否抛出特定异常及其消息
        with pytest.raises(ValueError, match=msg):
            # 调用 optimize.minimize 函数，传入目标函数 f、初始点 x0、优化方法和约束 bounds
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)

        # 创建另一个 Bounds 对象，指定不合法的上下界（某些上界小于对应的下界）
        bounds = Bounds([1, 6, 1], [3, 4, 2])
        # 设置错误消息，用于检查是否抛出 ValueError 异常
        msg = 'An upper bound is less than the corresponding lower bound.'
        # 使用 pytest 的上下文管理器检查是否抛出特定异常及其消息
        with pytest.raises(ValueError, match=msg):
            # 调用 optimize.minimize 函数，传入目标函数 f、初始点 x0、优化方法和约束 bounds
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)

    # 使用 pytest 的参数化装饰器，测试多个优化方法下的 optimize.minimize 函数的警告处理
    @pytest.mark.parametrize('method', ['bfgs', 'cg', 'newton-cg', 'powell'])
    def test_minimize_warnings_gh1953(self, method):
        # 定义一个 lambda 函数，计算 Rosenbrock 函数的梯度，除了 Powell 方法之外的其他方法都需要使用梯度
        kwargs = {} if method == 'powell' else {'jac': optimize.rosen_der}
        # 根据不同的方法选择不同的警告类型
        warning_type = (RuntimeWarning if method == 'powell'
                        else optimize.OptimizeWarning)

        # 设置 optimize.minimize 函数的选项，显示迭代过程和设置最大迭代次数
        options = {'disp': True, 'maxiter': 10}
        # 使用 pytest 的上下文管理器检查是否发出特定类型的警告及其消息
        with pytest.warns(warning_type, match='Maximum number'):
            # 调用 optimize.minimize 函数，传入 Rosenbrock 函数、初始点 [0, 0]、优化方法和选项
            optimize.minimize(lambda x: optimize.rosen(x), [0, 0],
                              method=method, options=options, **kwargs)

        # 更新选项，禁用迭代过程的显示
        options['disp'] = False
        # 调用 optimize.minimize 函数，传入 Rosenbrock 函数、初始点 [0, 0]、优化方法和更新后的选项
        optimize.minimize(lambda x: optimize.rosen(x), [0, 0],
                          method=method, options=options, **kwargs)
@pytest.mark.parametrize(
    'method',
    ['l-bfgs-b', 'tnc', 'Powell', 'Nelder-Mead', 'cobyqa']
)
# 定义测试函数 test_minimize_with_scalar，使用参数化测试方法，遍历不同优化方法
def test_minimize_with_scalar(method):
    # 定义目标函数 f(x)，计算 x 向量的平方和
    def f(x):
        return np.sum(x ** 2)

    # 调用 optimize.minimize 函数，传入目标函数 f，初始值为 17，设置边界 [-100, 100]，使用给定的优化方法
    res = optimize.minimize(f, 17, bounds=[(-100, 100)], method=method)
    # 断言优化成功
    assert res.success
    # 断言优化得到的最优解 res.x 接近 [0.0]
    assert_allclose(res.x, [0.0], atol=1e-5)


class TestLBFGSBBounds:
    def setup_method(self):
        # 设置测试环境，定义边界和预期解
        self.bounds = ((1, None), (None, None))
        self.solution = (1, 0)

    # 定义目标函数 fun，计算给定向量 x 的 p 次幂和
    def fun(self, x, p=2.0):
        return 1.0 / p * (x[0]**p + x[1]**p)

    # 定义雅可比函数 jac，计算给定向量 x 的 (p-1) 次幂
    def jac(self, x, p=2.0):
        return x**(p - 1)

    # 定义联合目标和雅可比函数 fj，返回 fun 和 jac 计算结果的元组
    def fj(self, x, p=2.0):
        return self.fun(x, p), self.jac(x, p)

    # 测试 L-BFGS-B 方法在给定边界下的表现
    def test_l_bfgs_b_bounds(self):
        x, f, d = optimize.fmin_l_bfgs_b(self.fun, [0, -1],
                                         fprime=self.jac,
                                         bounds=self.bounds)
        # 断言无警告标志并且优化结果 x 接近预期解 self.solution
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-6)

    # 测试使用 fun 和 jac 联合函数的 L-BFGS-B 方法在给定边界下的表现
    def test_l_bfgs_b_funjac(self):
        # 使用 optimize.fmin_l_bfgs_b 函数，传入联合函数 fj，初始值为 [0, -1]，额外参数为 (2.0,)
        x, f, d = optimize.fmin_l_bfgs_b(self.fj, [0, -1], args=(2.0, ),
                                         bounds=self.bounds)
        # 断言无警告标志并且优化结果 x 接近预期解 self.solution
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-6)

    # 测试使用 L-BFGS-B 方法进行最小化优化，带有边界设置
    def test_minimize_l_bfgs_b_bounds(self):
        res = optimize.minimize(self.fun, [0, -1], method='L-BFGS-B',
                                jac=self.jac, bounds=self.bounds)
        # 断言优化成功，并且优化结果 res.x 接近预期解 self.solution
        assert res['success'], res['message']
        assert_allclose(res.x, self.solution, atol=1e-6)

    @pytest.mark.parametrize('bounds', [
        ([(10, 1), (1, 10)]),
        ([(1, 10), (10, 1)]),
        ([(10, 1), (10, 1)])
    ])
    # 参数化测试：测试使用 L-BFGS-B 方法进行最小化优化时不正确的边界设置
    def test_minimize_l_bfgs_b_incorrect_bounds(self, bounds):
        with pytest.raises(ValueError, match='.*bound.*'):
            # 断言调用 optimize.minimize 函数时会抛出 ValueError 异常，匹配异常消息中包含 'bound'
            optimize.minimize(self.fun, [0, -1], method='L-BFGS-B',
                              jac=self.jac, bounds=bounds)

    # 测试使用 L-BFGS-B 方法进行最小化优化，带有边界设置和有限差分参数
    def test_minimize_l_bfgs_b_bounds_FD(self):
        # 遍历不同的有限差分和参数组合
        jacs = ['2-point', '3-point', None]
        argss = [(2.,), ()]
        for jac, args in itertools.product(jacs, argss):
            res = optimize.minimize(self.fun, [0, -1], args=args,
                                    method='L-BFGS-B',
                                    jac=jac, bounds=self.bounds,
                                    options={'finite_diff_rel_step': None})
            # 断言优化成功，并且优化结果 res.x 接近预期解 self.solution
            assert res['success'], res['message']
            assert_allclose(res.x, self.solution, atol=1e-6)


class TestOptimizeScalar:
    def setup_method(self):
        # 设置测试环境，定义预期解
        self.solution = 1.5
    # 定义一个目标函数，计算 (x - a)^2 - 0.8 的值
    def fun(self, x, a=1.5):
        """Objective function"""
        return (x - a)**2 - 0.8

    # 测试使用 optimize.brent 函数对目标函数进行优化
    def test_brent(self):
        # 使用默认参数调用 optimize.brent，并检查结果是否接近预期解 self.solution
        x = optimize.brent(self.fun)
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用指定的初始区间 brack 调用 optimize.brent，并检查结果是否接近预期解 self.solution
        x = optimize.brent(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 full_output=True 参数调用 optimize.brent，检查结果是否接近预期解 self.solution
        x = optimize.brent(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-6)

        # 使用多个初始区间 brack 调用 optimize.brent，并检查结果是否接近预期解 self.solution
        x = optimize.brent(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 pytest 检查 optimize.brent 抛出的 ValueError 是否符合指定的错误消息
        message = r"\(f\(xb\) < f\(xa\)\) and \(f\(xb\) < f\(xc\)\)"
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(-1, 0, 1))

        # 使用 pytest 检查 optimize.brent 抛出的 ValueError 是否符合指定的错误消息
        message = r"\(xa < xb\) and \(xb < xc\)"
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(0, -1, 1))

    # 使用 pytest 标记忽略 UserWarning，并测试 optimize.golden 函数
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_golden(self):
        # 使用 optimize.golden 函数优化目标函数，检查结果是否接近预期解 self.solution
        x = optimize.golden(self.fun)
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用指定的初始区间 brack 调用 optimize.golden，并检查结果是否接近预期解 self.solution
        x = optimize.golden(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 full_output=True 参数调用 optimize.golden，检查结果是否接近预期解 self.solution
        x = optimize.golden(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-6)

        # 使用多个初始区间 brack 调用 optimize.golden，并检查结果是否接近预期解 self.solution
        x = optimize.golden(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 tol=0 参数调用 optimize.golden，并检查结果是否接近预期解 self.solution
        x = optimize.golden(self.fun, tol=0)
        assert_allclose(x, self.solution)

        # 测试不同的 maxiter 参数对 optimize.golden 的影响
        maxiter_test_cases = [0, 1, 5]
        for maxiter in maxiter_test_cases:
            x0 = optimize.golden(self.fun, maxiter=0, full_output=True)
            x = optimize.golden(self.fun, maxiter=maxiter, full_output=True)
            nfev0, nfev = x0[2], x[2]
            assert_equal(nfev - nfev0, maxiter)

        # 使用 pytest 检查 optimize.golden 抛出的 ValueError 是否符合指定的错误消息
        message = r"\(f\(xb\) < f\(xa\)\) and \(f\(xb\) < f\(xc\)\)"
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(-1, 0, 1))

        # 使用 pytest 检查 optimize.golden 抛出的 ValueError 是否符合指定的错误消息
        message = r"\(xa < xb\) and \(xb < xc\)"
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(0, -1, 1))

    # 测试 optimize.fminbound 函数对目标函数进行优化
    def test_fminbound(self):
        # 使用 optimize.fminbound 函数优化目标函数，在区间 [0, 1] 内，检查结果是否接近预期解 1
        x = optimize.fminbound(self.fun, 0, 1)
        assert_allclose(x, 1, atol=1e-4)

        # 使用 optimize.fminbound 函数优化目标函数，在区间 [1, 5] 内，检查结果是否接近预期解 self.solution
        x = optimize.fminbound(self.fun, 1, 5)
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 optimize.fminbound 函数优化目标函数，传入 numpy 数组作为区间端点，检查结果是否接近预期解 self.solution
        x = optimize.fminbound(self.fun, np.array([1]), np.array([5]))
        assert_allclose(x, self.solution, atol=1e-6)

        # 使用 assert_raises 检查 optimize.fminbound 抛出的 ValueError
        assert_raises(ValueError, optimize.fminbound, self.fun, 5, 1)

    # 测试 optimize.fminbound 函数对目标函数进行优化，传入标量和数组作为参数
    def test_fminbound_scalar(self):
        # 使用 pytest 检查 optimize.fminbound 抛出的 ValueError 是否符合指定的错误消息
        with pytest.raises(ValueError, match='.*must be finite scalars.*'):
            optimize.fminbound(self.fun, np.zeros((1, 2)), 1)

        # 使用 optimize.fminbound 函数优化目标函数，在区间 [1, 5] 内，检查结果是否接近预期解 self.solution
        x = optimize.fminbound(self.fun, 1, np.array(5))
        assert_allclose(x, self.solution, atol=1e-6)

    # 测试 optimize.fminbound 函数对 fun(x)=x^2 函数进行优化，传入的区间为 [0, 0]，期望无输出
    def test_gh11207(self):
        def fun(x):
            return x**2
        optimize.fminbound(fun, 0, 0)
    def test_minimize_scalar(self):
        # 聚合所有上述测试用例，针对 minimize_scalar 包装器
        x = optimize.minimize_scalar(self.fun).x
        # 断言最小化标量函数的结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='Brent')
        # 使用 Brent 方法最小化标量函数，检查优化是否成功
        assert x.success

        x = optimize.minimize_scalar(self.fun, method='Brent',
                                     options=dict(maxiter=3))
        # 使用 Brent 方法最小化标量函数，指定最大迭代次数为3，检查优化是否失败
        assert not x.success

        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2),
                                     args=(1.5, ), method='Brent').x
        # 使用 Brent 方法最小化标量函数，在给定区间和参数的情况下，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='Brent',
                                     args=(1.5,)).x
        # 使用 Brent 方法最小化标量函数，传递额外参数，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15),
                                     args=(1.5, ), method='Brent').x
        # 使用 Brent 方法最小化标量函数，在给定区间和参数的情况下，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2),
                                     args=(1.5, ), method='golden').x
        # 使用黄金分割法最小化标量函数，在给定区间和参数的情况下，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='golden',
                                     args=(1.5,)).x
        # 使用黄金分割法最小化标量函数，传递额外参数，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15),
                                     args=(1.5, ), method='golden').x
        # 使用黄金分割法最小化标量函数，在给定区间和参数的情况下，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bounds=(0, 1), args=(1.5,),
                                     method='Bounded').x
        # 使用有界方法最小化标量函数，指定上下界和参数，断言结果接近1
        assert_allclose(x, 1, atol=1e-4)

        x = optimize.minimize_scalar(self.fun, bounds=(1, 5), args=(1.5, ),
                                     method='bounded').x
        # 使用有界方法最小化标量函数，指定上下界和参数，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bounds=(np.array([1]),
                                                       np.array([5])),
                                     args=(np.array([1.5]), ),
                                     method='bounded').x
        # 使用有界方法最小化标量函数，指定上下界和参数为数组，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)

        assert_raises(ValueError, optimize.minimize_scalar, self.fun,
                      bounds=(5, 1), method='bounded', args=(1.5, ))
        # 断言在错误的边界顺序下调用最小化标量函数会引发 ValueError

        assert_raises(ValueError, optimize.minimize_scalar, self.fun,
                      bounds=(np.zeros(2), 1), method='bounded', args=(1.5, ))
        # 断言在错误的边界顺序下调用最小化标量函数会引发 ValueError

        x = optimize.minimize_scalar(self.fun, bounds=(1, np.array(5)),
                                     method='bounded').x
        # 使用有界方法最小化标量函数，指定上界为数组，断言结果与预期解的近似性
        assert_allclose(x, self.solution, atol=1e-6)
    def test_minimize_scalar_custom(self):
        # 定义一个自定义最小化标量函数
        def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.1,
                    maxiter=100, callback=None, **options):
            # 初始最佳值为区间中点
            bestx = (bracket[1] + bracket[0]) / 2.0
            # 计算初始最佳值的函数值
            besty = fun(bestx)
            # 函数调用次数计数器
            funcalls = 1
            # 迭代次数计数器
            niter = 0
            # 是否有改进标志
            improved = True
            # 是否停止标志
            stop = False

            # 迭代优化过程
            while improved and not stop and niter < maxiter:
                # 重置改进标志
                improved = False
                # 迭代次数加一
                niter += 1
                # 在当前最佳值周围的两个点进行测试
                for testx in [bestx - stepsize, bestx + stepsize]:
                    # 计算测试点的函数值
                    testy = fun(testx, *args)
                    # 函数调用次数加一
                    funcalls += 1
                    # 如果找到更好的解，则更新最佳值和最佳函数值
                    if testy < besty:
                        besty = testy
                        bestx = testx
                        improved = True
                # 如果有回调函数，则调用
                if callback is not None:
                    callback(bestx)
                # 如果超过最大函数调用次数，则停止优化过程
                if maxfev is not None and funcalls >= maxfev:
                    stop = True
                    break

            # 返回优化结果对象
            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter,
                                           nfev=funcalls, success=(niter > 1))

        # 调用最小化标量函数的优化过程，使用自定义的custmin函数
        res = optimize.minimize_scalar(self.fun, bracket=(0, 4),
                                       method=custmin,
                                       options=dict(stepsize=0.05))
        # 断言最优解的精度
        assert_allclose(res.x, self.solution, atol=1e-6)

    def test_minimize_scalar_coerce_args_param(self):
        # 检测对参数args的强制类型转换，回归测试 gh-3503
        optimize.minimize_scalar(self.fun, args=1.5)

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_disp(self, method):
        # 测试所有最小化标量方法是否接受disp选项
        for disp in [0, 1, 2, 3]:
            optimize.minimize_scalar(self.fun, options={"disp": disp})

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_result_attributes(self, method):
        # 检查最小化标量结果对象是否包含特定属性
        kwargs = {"bounds": [-10, 10]} if method == 'bounded' else {}
        result = optimize.minimize_scalar(self.fun, method=method, **kwargs)
        assert hasattr(result, "x")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "fun")
        assert hasattr(result, "nfev")
        assert hasattr(result, "nit")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    # 测试处理包含 NaN 值的情况，确保程序能够正确退出
    def test_nan_values(self, method):
        # 设置随机种子确保结果可复现
        np.random.seed(1234)

        # 计数器，用于跟踪函数调用次数
        count = [0]

        # 定义一个函数，计数达到4次时返回 NaN，否则返回 x^2 + 0.1 * sin(x)
        def func(x):
            count[0] += 1
            if count[0] > 4:
                return np.nan
            else:
                return x**2 + 0.1 * np.sin(x)

        # 搜索的起始区间
        bracket = (-1, 0, 1)
        # 函数可接受的边界
        bounds = (-1, 1)

        # 忽略无效值的错误，使用上下文管理器压制特定的警告
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            # 过滤特定的警告类型，不向用户显示这些警告
            sup.filter(UserWarning, "delta_grad == 0.*")
            sup.filter(RuntimeWarning, ".*does not use Hessian.*")
            sup.filter(RuntimeWarning, ".*does not use gradient.*")

            # 重置计数器
            count = [0]

            # 根据方法选择是否设置边界参数
            kwargs = {"bounds": bounds} if method == 'bounded' else {}
            # 最小化单变量标量函数 func
            sol = optimize.minimize_scalar(func, bracket=bracket,
                                           **kwargs, method=method,
                                           options=dict(maxiter=20))
            # 断言优化失败
            assert_equal(sol.success, False)

    # 测试最小化标量函数的默认行为（gh-10911）
    def test_minimize_scalar_defaults_gh10911(self):
        # 定义一个简单的平方函数
        def f(x):
            return x**2

        # 最小化标量函数 f，默认起始点为 0
        res = optimize.minimize_scalar(f)
        # 断言最小化结果接近 0，允许一定的数值误差
        assert_allclose(res.x, 0, atol=1e-8)

        # 在指定的边界内最小化函数 f，期望最小值为 1
        res = optimize.minimize_scalar(f, bounds=(1, 100),
                                       options={'xatol': 1e-10})
        # 断言最小化结果为 1，允许一定的数值误差
        assert_allclose(res.x, 1)

    # 测试最小化标量函数时，边界包含非有限值的情况（gh-10911）
    def test_minimize_non_finite_bounds_gh10911(self):
        # 非有限边界会导致 ValueError 错误
        msg = "Optimization bounds must be finite scalars."
        # 测试函数 np.sin 在边界包含无穷大时是否触发 ValueError
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(1, np.inf))
        # 测试函数 np.sin 在边界包含 NaN 时是否触发 ValueError
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(np.nan, 1))

    # 使用参数化测试来检查最小化标量函数在不兼容的方法和边界设置时是否抛出错误（gh-10911）
    @pytest.mark.parametrize("method", ['brent', 'golden'])
    def test_minimize_unbounded_method_with_bounds_gh10911(self, method):
        # 当方法为 'brent' 或 'golden' 时，使用边界是不兼容的
        msg = "Use of `bounds` is incompatible with..."
        # 断言调用最小化标量函数 np.sin 时是否抛出 ValueError 错误
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, method=method, bounds=(1, 2))

    # 使用参数化测试来检查最小化标量函数的各种方法和容差设置下的行为
    @pytest.mark.parametrize("method", MINIMIZE_SCALAR_METHODS)
    @pytest.mark.parametrize("tol", [1, 1e-6])
    @pytest.mark.parametrize("fshape", [(), (1,), (1, 1)])
    # 定义测试函数，用于验证 `minimize_scalar` 方法在处理返回数组的目标函数时输出形状的一致性问题
    def test_minimize_scalar_dimensionality_gh16196(self, method, tol, fshape):
        # 定义目标函数，将输入的 x 的四次方转换为指定形状的 numpy 数组返回
        def f(x):
            return np.array(x**4).reshape(fshape)

        # 定义区间 a 和 b
        a, b = -0.1, 0.2
        # 根据方法类型选择不同的参数设置方式
        kwargs = (dict(bracket=(a, b)) if method != "bounded"
                  else dict(bounds=(a, b)))
        # 更新参数字典中的方法和容差设置
        kwargs.update(dict(method=method, tol=tol))

        # 调用 `minimize_scalar` 方法进行优化
        res = optimize.minimize_scalar(f, **kwargs)
        # 断言优化结果的 x、fun 属性以及目标函数 f(x) 的形状与预期形状 fshape 一致
        assert res.x.shape == res.fun.shape == f(res.x).shape == fshape

    # 使用 pytest 的参数化装饰器标记测试方法，验证 `minimize_scalar` 方法是否能够生成警告信息
    @pytest.mark.parametrize('method', ['bounded', 'brent', 'golden'])
    def test_minimize_scalar_warnings_gh1953(self, method):
        # 定义目标函数，计算 (x - 1)^2 的值
        def f(x):
            return (x - 1)**2

        # 根据方法类型选择不同的参数设置方式
        kwargs = {}
        kwd = 'bounds' if method == 'bounded' else 'bracket'
        kwargs[kwd] = [-2, 10]

        # 设置优化选项，其中 'disp' 为 True，'maxiter' 为 3
        options = {'disp': True, 'maxiter': 3}
        # 使用 pytest 的 warn 断言检查是否会触发优化警告，并匹配指定的警告信息
        with pytest.warns(optimize.OptimizeWarning, match='Maximum number'):
            optimize.minimize_scalar(f, method=method, options=options,
                                     **kwargs)

        # 将 'disp' 设置为 False，再次调用 `minimize_scalar` 方法进行优化
        options['disp'] = False
        optimize.minimize_scalar(f, method=method, options=options, **kwargs)
class TestBracket:

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_errors_and_status_false(self):
        # 定义一个函数 f(x)，用于测试 optimize.bracket 函数
        def f(x):  # gh-14858
            # 如果 x 在 (-1, 1) 范围内，则返回 x 的平方；否则返回 100.0
            return x**2 if ((-1 < x) & (x < 1)) else 100.0

        # 设置错误消息，用于检查 optimize.bracket 是否会抛出 RuntimeError 异常
        message = "The algorithm terminated without finding a valid bracket."
        
        # 测试 optimize.bracket 函数在不同输入下是否会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -1, 1)
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -1, np.inf)
        with pytest.raises(RuntimeError, match=message):
            optimize.brent(f, brack=(-1, 1))
        with pytest.raises(RuntimeError, match=message):
            optimize.golden(f, brack=(-1, 1))

        # 定义另一个函数 f(x)，用于测试 optimize.bracket 函数的迭代次数限制
        def f(x):  # gh-5899
            return -5 * x**5 + 4 * x**4 - 12 * x**3 + 11 * x**2 - 2 * x + 1

        # 设置错误消息，用于检查 optimize.bracket 在迭代次数限制下是否会抛出 RuntimeError 异常
        message = "No valid bracket was found before the iteration limit..."
        
        # 测试 optimize.bracket 函数在迭代次数有限的情况下是否会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -0.5, 0.5, maxiter=10)

    @pytest.mark.parametrize('method', ('brent', 'golden'))
    def test_minimize_scalar_success_false(self, method):
        # 定义一个函数 f(x)，用于测试 optimize.minimize_scalar 函数
        def f(x):  # gh-14858
            # 如果 x 在 (-1, 1) 范围内，则返回 x 的平方；否则返回 100.0
            return x**2 if ((-1 < x) & (x < 1)) else 100.0

        # 设置错误消息，用于检查 optimize.bracket 是否会抛出 RuntimeError 异常
        message = "The algorithm terminated without finding a valid bracket."

        # 使用 optimize.minimize_scalar 函数进行测试，检查返回结果的状态信息
        res = optimize.minimize_scalar(f, bracket=(-1, 1), method=method)
        assert not res.success
        assert message in res.message
        assert res.nfev == 3
        assert res.nit == 0
        assert res.fun == 100


def test_brent_negative_tolerance():
    # 检查 optimize.brent 函数在负的公差参数情况下是否会抛出 ValueError 异常
    assert_raises(ValueError, optimize.brent, np.cos, tol=-.01)


class TestNewtonCg:
    def test_rosenbrock(self):
        # 定义起始点 x0，并使用 optimize.minimize 函数进行 Rosenbrock 函数的最小化
        x0 = np.array([-1.2, 1.0])
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess=optimize.rosen_hess,
                                tol=1e-5,
                                method='Newton-CG')
        assert sol.success, sol.message
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)

    def test_himmelblau(self):
        # 使用 optimize.minimize 函数进行 Himmelblau 函数的最小化
        x0 = np.array(himmelblau_x0)
        sol = optimize.minimize(himmelblau,
                                x0,
                                jac=himmelblau_grad,
                                hess=himmelblau_hess,
                                method='Newton-CG',
                                tol=1e-6)
        assert sol.success, sol.message
        assert_allclose(sol.x, himmelblau_xopt, rtol=1e-4)
        assert_allclose(sol.fun, himmelblau_min, atol=1e-4)
    # 定义一个测试函数，用于测试有限差分法求解优化问题
    def test_finite_difference(self):
        # 初始化优化问题的起始点
        x0 = np.array([-1.2, 1.0])
        # 使用 Newton-CG 方法进行优化，最小化 Rosenbrock 函数
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess='2-point',  # 使用有限差分法计算 Hessian 矩阵
                                tol=1e-5,         # 设置收敛容差
                                method='Newton-CG')
        # 断言优化是否成功，并输出消息
        assert sol.success, sol.message
        # 检查优化后的解是否接近预期的最优解 [1, 1]
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)
    
    # 定义另一个测试函数，用于测试 Hessian 更新策略的影响
    def test_hessian_update_strategy(self):
        # 初始化优化问题的起始点
        x0 = np.array([-1.2, 1.0])
        # 使用 Newton-CG 方法进行优化，最小化 Rosenbrock 函数
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess=optimize.BFGS(),  # 使用 BFGS 方法计算 Hessian 矩阵
                                tol=1e-5,              # 设置收敛容差
                                method='Newton-CG')
        # 断言优化是否成功，并输出消息
        assert sol.success, sol.message
        # 检查优化后的解是否接近预期的最优解 [1, 1]
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)
# 定义一个测试函数 test_linesearch_powell，用于测试 optimize.py 中的非公共函数 _linesearch_powell
def test_linesearch_powell():
    # 从 optimize._optimize 模块中导入 _linesearch_powell 函数并赋值给 linesearch_powell
    linesearch_powell = optimize._optimize._linesearch_powell
    # 定义一个用于优化的测试函数 func，返回向量 x 与 [-1.0, 2.0, 1.5, -0.4] 的平方差之和
    def func(x):
        return np.sum((x - np.array([-1.0, 2.0, 1.5, -0.4])) ** 2)
    # 初始点 p0 设置为 [0., 0, 0, 0]
    p0 = np.array([0., 0, 0, 0])
    # 计算函数在 p0 处的值
    fval = func(p0)
    # 定义下界和上界分别为 [-∞, -∞, -∞, -∞] 和 [∞, ∞, ∞, ∞]
    lower_bound = np.array([-np.inf] * 4)
    upper_bound = np.array([np.inf] * 4)
    # 定义包含所有测试用例的元组
    all_tests = (
        (np.array([1., 0, 0, 0]), -1),    # 第一个测试：输入向量为 [1., 0, 0, 0]，期望结果为 -1
        (np.array([0., 1, 0, 0]), 2),     # 第二个测试：输入向量为 [0., 1, 0, 0]，期望结果为 2
        (np.array([0., 0, 1, 0]), 1.5),   # 第三个测试：输入向量为 [0., 0, 1, 0]，期望结果为 1.5
        (np.array([0., 0, 0, 1]), -.4),   # 第四个测试：输入向量为 [0., 0, 0, 1]，期望结果为 -0.4
        (np.array([-1., 0, 1, 0]), 1.25), # 第五个测试：输入向量为 [-1., 0, 1, 0]，期望结果为 1.25
        (np.array([0., 0, 1, 1]), .55),   # 第六个测试：输入向量为 [0., 0, 1, 1]，期望结果为 0.55
        (np.array([2., 0, -1, 1]), -.65), # 第七个测试：输入向量为 [2., 0, -1, 1]，期望结果为 -0.65
    )
    
    # 遍历所有测试用例
    for xi, l in all_tests:
        # 进行线搜索并返回结果
        f, p, direction = linesearch_powell(func, p0, xi,
                                            fval=fval, tol=1e-5)
        # 断言函数值应接近期望函数值
        assert_allclose(f, func(l * xi), atol=1e-6)
        # 断言搜索到的最优点应接近期望的乘积
        assert_allclose(p, l * xi, atol=1e-6)
        # 断言搜索方向应接近期望的乘积方向
        assert_allclose(direction, l * xi, atol=1e-6)
    
        # 再次进行线搜索并返回结果，带有额外的参数
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言函数值应接近期望函数值
        assert_allclose(f, func(l * xi), atol=1e-6)
        # 断言搜索到的最优点应接近期望的乘积
        assert_allclose(p, l * xi, atol=1e-6)
        # 断言搜索方向应接近期望的乘积方向
        assert_allclose(direction, l * xi, atol=1e-6)
# 定义一个测试函数，用于测试 optimize.py 中未公开的 linesearch_powell 函数。
def test_linesearch_powell_bounded():
    # 导入 linesearch_powell 函数，该函数位于 optimize.py 的 _optimize 模块中
    linesearch_powell = optimize._optimize._linesearch_powell
    
    # 定义一个简单的目标函数 func，计算给定向量与 [-1.0, 2.0, 1.5, -0.4] 的差的平方和
    def func(x):
        return np.sum((x - np.array([-1.0, 2.0, 1.5, -0.4])) ** 2)
    
    # 初始化优化的起始点 p0
    p0 = np.array([0., 0, 0, 0])
    # 计算目标函数在 p0 处的函数值 fval
    fval = func(p0)

    # 设置变量 lower_bound 和 upper_bound，定义问题的边界条件
    lower_bound = np.array([-2.]*4)
    upper_bound = np.array([2.]*4)

    # 定义一组测试用例 all_tests，每个测试用例包含一个方向向量 xi 和预期的最优参数 l
    all_tests = (
        (np.array([1., 0, 0, 0]), -1),
        (np.array([0., 1, 0, 0]), 2),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), 1.25),
        (np.array([0., 0, 1, 1]), .55),
        (np.array([2., 0, -1, 1]), -.65),
    )

    # 遍历所有测试用例，测试 linesearch_powell 函数的行为
    for xi, l in all_tests:
        # 调用 linesearch_powell 函数进行优化
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言优化后的函数值 f 与预期函数值 func(l * xi) 接近
        assert_allclose(f, func(l * xi), atol=1e-6)
        # 断言优化后的参数 p 与预期参数 l * xi 接近
        assert_allclose(p, l * xi, atol=1e-6)
        # 断言优化后的方向 direction 与预期方向 l * xi 接近
        assert_allclose(direction, l * xi, atol=1e-6)

    # 更新 lower_bound 和 upper_bound，设置新的边界条件以测试不同的情况
    lower_bound = np.array([-.3]*3 + [-1])
    upper_bound = np.array([.45]*3 + [.9])

    # 重新定义 all_tests，包含新的测试用例
    all_tests = (
        (np.array([1., 0, 0, 0]), -.3),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), .45),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), .3),
        (np.array([0., 0, 1, 1]), .45),
        (np.array([2., 0, -1, 1]), -.15),
    )

    # 再次遍历所有测试用例，测试更新后的边界条件下的 linesearch_powell 函数行为
    for xi, l in all_tests:
        # 调用 linesearch_powell 函数进行优化
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言优化后的函数值 f 与预期函数值 func(l * xi) 接近
        assert_allclose(f, func(l * xi), atol=1e-6)
        # 断言优化后的参数 p 与预期参数 l * xi 接近
        assert_allclose(p, l * xi, atol=1e-6)
        # 断言优化后的方向 direction 与预期方向 l * xi 接近
        assert_allclose(direction, l * xi, atol=1e-6)

    # 更新 p0 和 fval，以及 all_tests，测试起始点在边界之外的情况
    p0 = np.array([-1., 0, 0, 2])
    fval = func(p0)

    all_tests = (
        (np.array([1., 0, 0, 0]), .7),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), .45),
        (np.array([0., 0, 0, 1]), -2.4),
    )
    # 遍历所有测试数据对(xi, l)，执行以下操作
    for xi, l in all_tests:
        # 调用 linesearch_powell 函数进行 Powell 方法的线性搜索
        # func: 目标函数
        # p0: 初始点
        # xi: 搜索方向
        # tol: 容差值
        # lower_bound: 下界限制
        # upper_bound: 上界限制
        # fval: 目标函数在初始点的值
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言目标函数在搜索点 p 处的值近似等于 p0 + l * xi 处的目标函数值
        assert_allclose(f, func(p0 + l * xi), atol=1e-6)
        # 断言搜索结果 p 近似等于 p0 + l * xi
        assert_allclose(p, p0 + l * xi, atol=1e-6)
        # 断言搜索方向 direction 近似等于 l * xi
        assert_allclose(direction, l * xi, atol=1e-6)

    # 现在混合无穷大值进入测试
    p0 = np.array([0., 0, 0, 0])
    fval = func(p0)

    # 选择包含无穷大值的边界
    lower_bound = np.array([-.3, -np.inf, -np.inf, -1])
    upper_bound = np.array([np.inf, .45, np.inf, .9])

    # 定义包含不同测试点(xi, l)的测试集合
    all_tests = (
        (np.array([1., 0, 0, 0]), -.3),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), .3),
        (np.array([0., 0, 1, 1]), .55),
        (np.array([2., 0, -1, 1]), -.15),
    )

    # 遍历所有测试数据(xi, l)，执行以下操作
    for xi, l in all_tests:
        # 调用 linesearch_powell 函数进行 Powell 方法的线性搜索
        # func: 目标函数
        # p0: 初始点
        # xi: 搜索方向
        # tol: 容差值
        # lower_bound: 下界限制
        # upper_bound: 上界限制
        # fval: 目标函数在初始点的值
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言目标函数在搜索点 p 处的值近似等于 l * xi 处的目标函数值
        assert_allclose(f, func(l * xi), atol=1e-6)
        # 断言搜索结果 p 近似等于 l * xi
        assert_allclose(p, l * xi, atol=1e-6)
        # 断言搜索方向 direction 近似等于 l * xi
        assert_allclose(direction, l * xi, atol=1e-6)

    # 选择包含无穷大值的边界
    lower_bound = np.array([-.3, -np.inf, -np.inf, -1])
    upper_bound = np.array([np.inf, .45, np.inf, .9])

    # 选择初始点在边界之外的情况
    p0 = np.array([-1., 0, 0, 2])
    fval = func(p0)

    # 定义包含不同测试点(xi, l)的测试集合
    all_tests = (
        (np.array([1., 0, 0, 0]), .7),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -2.4),
    )

    # 遍历所有测试数据(xi, l)，执行以下操作
    for xi, l in all_tests:
        # 调用 linesearch_powell 函数进行 Powell 方法的线性搜索
        # func: 目标函数
        # p0: 初始点
        # xi: 搜索方向
        # tol: 容差值
        # lower_bound: 下界限制
        # upper_bound: 上界限制
        # fval: 目标函数在初始点的值
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        # 断言目标函数在搜索点 p 处的值近似等于 p0 + l * xi 处的目标函数值
        assert_allclose(f, func(p0 + l * xi), atol=1e-6)
        # 断言搜索结果 p 近似等于 p0 + l * xi
        assert_allclose(p, p0 + l * xi, atol=1e-6)
        # 断言搜索方向 direction 近似等于 l * xi
        assert_allclose(direction, l * xi, atol=1e-6)
def test_powell_limits():
    # 修复问题 gh15342 - Powell 方法在某些函数评估中超出边界的情况
    bounds = optimize.Bounds([0, 0], [0.6, 20])

    def fun(x):
        a, b = x
        # 断言确保 x 在定义的边界内
        assert (x >= bounds.lb).all() and (x <= bounds.ub).all()
        return a ** 2 + b ** 2

    # 使用 Powell 方法最小化函数 fun，初始点为 [0.6, 20]，并应用边界限制
    optimize.minimize(fun, x0=[0.6, 20], method='Powell', bounds=bounds)

    # 另一个来自原始报告的测试 - gh-13411
    bounds = optimize.Bounds(lb=[0,], ub=[1,], keep_feasible=[True,])

    def func(x):
        # 断言确保 x 在 [0, 1] 的范围内
        assert x >= 0 and x <= 1
        return np.exp(x)

    # 使用 Powell 方法最小化函数 func，初始点为 [0.5]，并应用边界限制
    optimize.minimize(fun=func, x0=[0.5], method='powell', bounds=bounds)


class TestRosen:

    def test_hess(self):
        # 比较 rosen_hess(x) 与 rosen_hess_prod(x, p) 的点积，参见 gh-1775
        x = np.array([3, 4, 5])
        p = np.array([2, 2, 2])
        hp = optimize.rosen_hess_prod(x, p)
        dothp = np.dot(optimize.rosen_hess(x), p)
        # 断言确保两者相等
        assert_equal(hp, dothp)


def himmelblau(p):
    """
    R^2 -> R^1 的优化测试函数。该函数有四个局部最小值，其中 himmelblau(xopt) == 0。
    """
    x, y = p
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b


def himmelblau_grad(p):
    x, y = p
    return np.array([4*x**3 + 4*x*y - 42*x + 2*y**2 - 14,
                     2*x**2 + 4*x*y + 4*y**3 - 26*y - 22])


def himmelblau_hess(p):
    x, y = p
    return np.array([[12*x**2 + 4*y - 42, 4*x + 4*y],
                     [4*x + 4*y, 4*x + 12*y**2 - 26]])


himmelblau_x0 = [-0.27, -0.9]
himmelblau_xopt = [3, 2]
himmelblau_min = 0.0


def test_minimize_multiple_constraints():
    # gh-4240 的回归测试
    def func(x):
        return np.array([25 - 0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])

    def func1(x):
        return np.array([x[1]])

    def func2(x):
        return np.array([x[2]])

    # 定义不等式约束
    cons = ({'type': 'ineq', 'fun': func},
            {'type': 'ineq', 'fun': func1},
            {'type': 'ineq', 'fun': func2})

    def f(x):
        return -1 * (x[0] + x[1] + x[2])

    # 使用 SLSQP 方法最小化目标函数 f，初始点为 [0, 0, 0]，并应用约束条件 cons
    res = optimize.minimize(f, [0, 0, 0], method='SLSQP', constraints=cons)
    # 断言确保最小化的结果接近预期解 [125, 0, 0]
    assert_allclose(res.x, [125, 0, 0], atol=1e-10)


class TestOptimizeResultAttributes:
    # 测试所有最小化器返回的 OptimizeResult 包含所有 OptimizeResult 属性
    def setup_method(self):
        self.x0 = [5, 5]
        self.func = optimize.rosen
        self.jac = optimize.rosen_der
        self.hess = optimize.rosen_hess
        self.hessp = optimize.rosen_hess_prod
        self.bounds = [(0., 10.), (0., 10.)]

    @pytest.mark.fail_slow(2)
    # 测试函数，用于验证优化结果对象中是否包含指定的属性
    def test_attributes_present(self):
        # 定义需要验证的属性列表
        attributes = ['nit', 'nfev', 'x', 'success', 'status', 'fun', 'message']
        # 定义跳过某些方法和属性的映射关系
        skip = {'cobyla': ['nit']}
        
        # 遍历所有优化方法
        for method in MINIMIZE_METHODS:
            # 使用 suppress_warnings 上下文管理器，捕获运行时警告
            with suppress_warnings() as sup:
                # 过滤特定的运行时警告，这些警告表明某些方法不使用梯度或海森信息
                sup.filter(RuntimeWarning,
                           ("Method .+ does not use (gradient|Hessian.*)"
                            " information"))
                
                # 执行优化过程，调用 optimize.minimize 函数
                res = optimize.minimize(self.func, self.x0, method=method,
                                        jac=self.jac, hess=self.hess,
                                        hessp=self.hessp)
            
            # 验证每个属性是否存在于优化结果对象中
            for attribute in attributes:
                # 如果当前方法在跳过列表中，并且当前属性也在跳过方法的属性列表中，则跳过当前属性的验证
                if method in skip and attribute in skip[method]:
                    continue
                
                # 使用断言检查属性是否存在于结果对象中
                assert hasattr(res, attribute)
                # 使用断言检查属性是否在结果对象的属性列表中
                assert attribute in dir(res)
            
            # 验证特定问题修复，确保 OptimizeResult.message 是一个字符串类型
            # gh13001, OptimizeResult.message 应该是一个字符串
            assert isinstance(res.message, str)
# 定义函数 f1，接受一个点 z 和一系列参数 params
def f1(z, *params):
    # 解包点 z 为 x 和 y
    x, y = z
    # 解包参数 params，依次为 a, b, c, d, e, f, g, h, i, j, k, l, scale
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    # 返回 f1 函数的计算结果
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)


# 定义函数 f2，接受一个点 z 和一系列参数 params
def f2(z, *params):
    # 解包点 z 为 x 和 y
    x, y = z
    # 解包参数 params，依次为 a, b, c, d, e, f, g, h, i, j, k, l, scale
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    # 返回 f2 函数的计算结果
    return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))


# 定义函数 f3，接受一个点 z 和一系列参数 params
def f3(z, *params):
    # 解包点 z 为 x 和 y
    x, y = z
    # 解包参数 params，依次为 a, b, c, d, e, f, g, h, i, j, k, l, scale
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    # 返回 f3 函数的计算结果
    return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))


# 定义函数 brute_func，接受一个点 z 和一系列参数 params
def brute_func(z, *params):
    # 调用 f1、f2 和 f3 函数，并返回它们的总和
    return f1(z, *params) + f2(z, *params) + f3(z, *params)


# 定义一个测试类 TestBrute
class TestBrute:
    # 设置方法，初始化测试所需的参数和解决方案
    def setup_method(self):
        self.params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
        self.rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
        self.solution = np.array([-1.05665192, 1.80834843])

    # 实例方法 brute_func，用于优化
    def brute_func(self, z, *params):
        return brute_func(z, *params)

    # 测试 brute 方法
    def test_brute(self):
        # 使用 optimize.brute 方法进行优化，使用 finish=optimize.fmin 作为后处理器
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True, finish=optimize.fmin)
        # 断言优化结果与预期解决方案在给定的容差范围内相等
        assert_allclose(resbrute[0], self.solution, atol=1e-3)
        # 断言优化结果的函数值与预期解决方案在给定的容差范围内相等
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params),
                        atol=1e-3)

        # 使用 optimize.brute 方法进行优化，使用 finish=optimize.minimize 作为后处理器
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True,
                                  finish=optimize.minimize)
        # 断言优化结果与预期解决方案在给定的容差范围内相等
        assert_allclose(resbrute[0], self.solution, atol=1e-3)
        # 断言优化结果的函数值与预期解决方案在给定的容差范围内相等
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params),
                        atol=1e-3)

        # 测试 optimize.brute 是否可以优化一个实例方法（其他测试使用非类方法的函数）
        resbrute = optimize.brute(self.brute_func, self.rranges,
                                  args=self.params, full_output=True,
                                  finish=optimize.minimize)
        # 断言优化结果与预期解决方案在给定的容差范围内相等
        assert_allclose(resbrute[0], self.solution, atol=1e-3)

    # 测试一维问题
    def test_1D(self):
        # 测试对于一维问题，测试函数接收一个数组而不是标量
        def f(x):
            assert len(x.shape) == 1
            assert x.shape[0] == 1
            return x ** 2

        # 使用 optimize.brute 方法进行优化，不使用后处理器
        optimize.brute(f, [(-1, 1)], Ns=3, finish=None)

    # 测试工作线程功能
    @pytest.mark.fail_slow(10)
    def test_workers(self):
        # 检查并行评估是否工作
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True, finish=None)

        resbrute1 = optimize.brute(brute_func, self.rranges, args=self.params,
                                   full_output=True, finish=None, workers=2)

        # 断言并行计算的结果与单线程计算的结果在给定的容差范围内相等
        assert_allclose(resbrute1[-1], resbrute[-1])
        assert_allclose(resbrute1[0], resbrute[0])
    # 定义一个测试方法，用于测试 RuntimeWarning 的捕获情况，使用 capsys 参数来捕获输出
    def test_runtime_warning(self, capsys):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(1234)

        # 定义一个函数 func，接受 z 和任意数量的参数，返回一个随机数乘以 1000
        def func(z, *params):
            return rng.random(1) * 1000  # 这里的问题是永远不收敛

        # 定义一个匹配的警告消息
        msg = "final optimization did not succeed.*|Maximum number of function eval.*"
        
        # 使用 pytest 的 warn 函数来捕获 RuntimeWarning，匹配指定的消息
        with pytest.warns(RuntimeWarning, match=msg):
            # 调用 optimize.brute 方法，并传入 func 函数、参数范围 self.rranges 和 self.params，显示过程中的详细信息
            optimize.brute(func, self.rranges, args=self.params, disp=True)

    # 定义一个测试方法，用于测试 optimize.brute 是否能够强制将非可迭代参数转换为元组
    def test_coerce_args_param(self):
        # 定义一个函数 f，接受 x 和任意数量的参数 args，返回 x 的 args[0] 次幂
        def f(x, *args):
            return x ** args[0]

        # 调用 optimize.brute 方法，传入函数 f，参数范围为 (slice(-4, 4, .25),)，并将参数 args 设置为 2
        resbrute = optimize.brute(f, (slice(-4, 4, .25),), args=2)
        
        # 使用 assert_allclose 断言方法来检查 resbrute 是否接近于 0
        assert_allclose(resbrute, 0)
@pytest.mark.fail_slow(20)
# 使用 pytest 的装饰器标记此测试函数为失败慢测试，最长运行时间为20秒
def test_cobyla_threadsafe():

    # Verify that cobyla is threadsafe. Will segfault if it is not.
    # 验证 COBYLA 方法在多线程环境下是否安全。如果不安全将导致段错误。

    import concurrent.futures
    import time

    def objective1(x):
        # Objective function 1: squared value of x[0]
        # 目标函数1：计算 x[0] 的平方
        time.sleep(0.1)
        return x[0]**2

    def objective2(x):
        # Objective function 2: squared difference from 1 of x[0]
        # 目标函数2：计算 x[0] 与 1 的差的平方
        time.sleep(0.1)
        return (x[0]-1)**2

    min_method = "COBYLA"

    def minimizer1():
        # Minimize objective1 using COBYLA method
        # 使用 COBYLA 方法最小化目标函数1
        return optimize.minimize(objective1,
                                      [0.0],
                                      method=min_method)

    def minimizer2():
        # Minimize objective2 using COBYLA method
        # 使用 COBYLA 方法最小化目标函数2
        return optimize.minimize(objective2,
                                      [0.0],
                                      method=min_method)

    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = []
        tasks.append(pool.submit(minimizer1))
        tasks.append(pool.submit(minimizer2))
        for t in tasks:
            t.result()


class TestIterationLimits:
    # Tests that optimisation does not give up before trying requested
    # number of iterations or evaluations. And that it does not succeed
    # by exceeding the limits.
    # 测试优化过程在不放弃所请求的迭代次数或评估次数前是否能完成，并且不会在超过限制时成功。

    def setup_method(self):
        # Setup method to initialize funcalls counter
        # 设置方法，初始化 funcalls 计数器
        self.funcalls = 0

    def slow_func(self, v):
        # Function for slow computation
        # 用于耗时计算的函数
        self.funcalls += 1
        r, t = np.sqrt(v[0]**2+v[1]**2), np.arctan2(v[0], v[1])
        return np.sin(r*20 + t)+r*0.5

    @pytest.mark.fail_slow(10)
    # 使用 pytest 的装饰器标记此测试函数为失败慢测试，最长运行时间为10秒
    def test_neldermead_limit(self):
        # Test Nelder-Mead optimization with specified iteration limit
        # 测试使用指定迭代限制的 Nelder-Mead 优化方法
        self.check_limits("Nelder-Mead", 200)

    def test_powell_limit(self):
        # Test Powell optimization with specified iteration limit
        # 测试使用指定迭代限制的 Powell 优化方法
        self.check_limits("powell", 1000)
    # 定义一个方法来检查优化方法的限制条件和结果
    def check_limits(self, method, default_iters):
        # 遍历三组初始值列表，分别是 [0.1, 0.1], [1, 1], [2, 2]
        for start_v in [[0.1, 0.1], [1, 1], [2, 2]]:
            # 遍历不同的最大函数评估次数 mfev
            for mfev in [50, 500, 5000]:
                # 初始化函数调用次数为 0
                self.funcalls = 0
                # 使用 optimize.minimize 函数优化 self.slow_func 函数
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxfev": mfev})
                # 断言函数实际调用次数与结果中记录的函数评估次数相同
                assert self.funcalls == res["nfev"]
                # 如果优化成功，断言函数评估次数小于 mfev
                if res["success"]:
                    assert res["nfev"] < mfev
                else:
                    # 如果优化失败，断言函数评估次数大于等于 mfev
                    assert res["nfev"] >= mfev

            # 遍历不同的最大迭代次数 mit
            for mit in [50, 500, 5000]:
                # 再次使用 optimize.minimize 函数优化 self.slow_func 函数
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit})
                # 如果优化成功，断言迭代次数小于等于 mit
                if res["success"]:
                    assert res["nit"] <= mit
                else:
                    # 如果优化失败，断言迭代次数大于等于 mit
                    assert res["nit"] >= mit

            # 遍历同时设置最大函数评估次数 mfev 和最大迭代次数 mit 的组合
            for mfev, mit in [[50, 50], [5000, 5000], [5000, np.inf]]:
                # 初始化函数调用次数为 0
                self.funcalls = 0
                # 使用 optimize.minimize 函数优化 self.slow_func 函数
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                # 断言函数实际调用次数与结果中记录的函数评估次数相同
                assert self.funcalls == res["nfev"]
                # 如果优化成功，断言函数评估次数小于 mfev 且迭代次数小于等于 mit
                if res["success"]:
                    assert res["nfev"] < mfev and res["nit"] <= mit
                else:
                    # 如果优化失败，断言函数评估次数大于等于 mfev 或迭代次数大于等于 mit
                    assert res["nfev"] >= mfev or res["nit"] >= mit

            # 遍历最大函数评估次数 mfev 或最大迭代次数 mit 为无限的情况
            for mfev, mit in [[np.inf, None], [None, np.inf]]:
                # 初始化函数调用次数为 0
                self.funcalls = 0
                # 使用 optimize.minimize 函数优化 self.slow_func 函数
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                # 断言函数实际调用次数与结果中记录的函数评估次数相同
                assert self.funcalls == res["nfev"]
                # 如果优化成功
                if res["success"]:
                    # 如果 mfev 为 None，断言函数评估次数小于默认迭代次数的两倍
                    if mfev is None:
                        assert res["nfev"] < default_iters*2
                    else:
                        # 否则，断言迭代次数小于等于默认迭代次数的两倍
                        assert res["nit"] <= default_iters*2
                else:
                    # 如果优化失败，断言函数评估次数大于等于默认迭代次数的两倍或迭代次数大于等于默认迭代次数的两倍
                    assert (res["nfev"] >= default_iters*2
                            or res["nit"] >= default_iters*2)
def test_result_x_shape_when_len_x_is_one():
    def fun(x):
        return x * x  # 定义一个计算平方的函数

    def jac(x):
        return 2. * x  # 定义该函数的雅可比矩阵（梯度）

    def hess(x):
        return np.array([[2.]])  # 定义该函数的黑塞矩阵（海森矩阵）

    # 定义一组优化方法
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC',
               'COBYLA', 'COBYQA', 'SLSQP']
    # 对于每种方法，执行优化并验证结果形状
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method)
        assert res.x.shape == (1,)

    # 使用带有雅可比和黑塞矩阵的优化方法
    methods = ['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
               'trust-krylov', 'Newton-CG']
    # 对于每种方法，执行带有雅可比和黑塞矩阵的优化，并验证结果形状
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method, jac=jac,
                                hess=hess)
        assert res.x.shape == (1,)


class FunctionWithGradient:
    def __init__(self):
        self.number_of_calls = 0  # 初始化函数调用次数为0

    def __call__(self, x):
        self.number_of_calls += 1  # 每次调用增加函数调用次数
        return np.sum(x**2), 2 * x  # 返回函数值和梯度值


@pytest.fixture
def function_with_gradient():
    return FunctionWithGradient()  # 返回一个FunctionWithGradient实例


def test_memoize_jac_function_before_gradient(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function(x0), 5.0)  # 测试记忆化函数的功能
    assert function_with_gradient.number_of_calls == 1  # 验证函数调用次数

    assert_allclose(memoized_function.derivative(x0), 2 * x0)  # 测试记忆化函数的梯度计算
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if gradient is requested after function value"  # 验证如果先请求函数值后请求梯度，函数是否被重新计算

    assert_allclose(
        memoized_function(2 * x0), 20.0,
        err_msg="different input triggers new computation")  # 测试不同的输入是否会触发新的计算
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_memoize_jac_gradient_before_function(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function.derivative(x0), 2 * x0)  # 测试记忆化函数的梯度计算
    assert function_with_gradient.number_of_calls == 1  # 验证函数调用次数

    assert_allclose(memoized_function(x0), 5.0)  # 测试记忆化函数的功能
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if function value is requested after gradient"  # 验证如果先请求梯度后请求函数值，函数是否被重新计算

    assert_allclose(
        memoized_function.derivative(2 * x0), 4 * x0,
        err_msg="different input triggers new computation")  # 测试不同的输入是否会触发新的计算
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_memoize_jac_with_bfgs(function_with_gradient):
    """ Tests that using MemoizedJac in combination with ScalarFunction
        and BFGS does not lead to repeated function evaluations.
        Tests changes made in response to GH11868.
    """
    memoized_function = MemoizeJac(function_with_gradient)
    jac = memoized_function.derivative
    hess = optimize.BFGS()

    x0 = np.array([1.0, 0.5])
    scalar_function = ScalarFunction(
        memoized_function, x0, (), jac, hess, None, None)
    # 断言，验证 function_with_gradient.number_of_calls 等于 1
    assert function_with_gradient.number_of_calls == 1
    
    # 调用 scalar_function.fun 方法，参数为 x0 + 0.1，并增加 function_with_gradient.number_of_calls 计数
    scalar_function.fun(x0 + 0.1)
    # 断言，验证 function_with_gradient.number_of_calls 等于 2
    assert function_with_gradient.number_of_calls == 2
    
    # 再次调用 scalar_function.fun 方法，参数为 x0 + 0.2，并增加 function_with_gradient.number_of_calls 计数
    scalar_function.fun(x0 + 0.2)
    # 断言，验证 function_with_gradient.number_of_calls 等于 3
    assert function_with_gradient.number_of_calls == 3
# 定义一个测试函数，用于测试 issue gh-12696，验证 optimize.fminbound 是否会抛出警告
def test_gh12696():
    # 使用 assert_no_warnings 上下文管理器确保 optimize.fminbound 不会抛出警告
    with assert_no_warnings():
        # 调用 optimize.fminbound 函数进行优化，传入 lambda 表达式作为目标函数，不显示优化过程
        optimize.fminbound(
            lambda x: np.array([x**2]), -np.pi, np.pi, disp=False)


# --- Test minimize with equal upper and lower bounds --- #

# 设置测试数据和函数
def setup_test_equal_bounds():
    # 固定随机种子，以便结果可重现
    np.random.seed(0)
    # 生成随机初始点 x0
    x0 = np.random.rand(4)
    # 设置下界 lb 和上界 ub
    lb = np.array([0, 2, -1, -1.0])
    ub = np.array([3, 2, 2, -1.0])
    # 标记 lb == ub 的位置
    i_eb = (lb == ub)

    # 定义检查 x 是否满足条件的函数
    def check_x(x, check_size=True, check_values=True):
        if check_size:
            assert x.size == 4
        if check_values:
            # 使用 assert_allclose 检查 x[i_eb] 是否与 lb[i_eb] 接近
            assert_allclose(x[i_eb], lb[i_eb])

    # 定义目标函数 func，使用 Rosenbrock 函数
    def func(x):
        check_x(x)
        return optimize.rosen(x)

    # 定义梯度函数 grad，计算 Rosenbrock 函数的梯度
    def grad(x):
        check_x(x)
        return optimize.rosen_der(x)

    # 定义回调函数 callback，用于每次迭代后检查 x 是否满足条件
    def callback(x, *args):
        check_x(x)

    # 定义约束条件 constraint1 和其雅可比矩阵 jacobian1
    def constraint1(x):
        check_x(x, check_values=False)
        return x[0:1] - 1

    def jacobian1(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[0] = 1
        return dc

    # 定义约束条件 constraint2 和其雅可比矩阵 jacobian2
    def constraint2(x):
        check_x(x, check_values=False)
        return x[2:3] - 0.5

    def jacobian2(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[2] = 1
        return dc

    # 创建 NonlinearConstraint 对象 c1a, c1b, c2a, c2b 分别对应不同的约束条件
    c1a = NonlinearConstraint(constraint1, -np.inf, 0)
    c1b = NonlinearConstraint(constraint1, -np.inf, 0, jacobian1)
    c2a = NonlinearConstraint(constraint2, -np.inf, 0)
    c2b = NonlinearConstraint(constraint2, -np.inf, 0, jacobian2)

    # 定义方法集合 methods，包括 L-BFGS-B, SLSQP, TNC
    methods = ('L-BFGS-B', 'SLSQP', 'TNC')

    # 定义关键字参数集合 kwds，分别测试无梯度、有梯度、联合目标/梯度函数的情况
    kwds = ({"fun": func, "jac": False},
            {"fun": func, "jac": grad},
            {"fun": (lambda x: (func(x), grad(x))),
             "jac": True})

    # 定义边界类型集合 bound_types，包括老式和新式边界约束方式
    bound_types = (lambda lb, ub: list(zip(lb, ub)),
                   Bounds)

    # 定义约束集合 constraints，包括不同组合的约束条件和其参考条件
    constraints = ((None, None), ([], []),
                   (c1a, c1b), (c2b, c2b),
                   ([c1b], [c1b]), ([c2a], [c2b]),
                   ([c1a, c2a], [c1b, c2b]),
                   ([c1a, c2b], [c1b, c2b]),
                   ([c1b, c2b], [c1b, c2b]))

    # 定义回调函数集合 callbacks，测试是否使用回调函数
    callbacks = (None, callback)

    # 构造数据字典 data，包含所有测试相关的数据和函数
    data = {"methods": methods, "kwds": kwds, "bound_types": bound_types,
            "constraints": constraints, "callbacks": callbacks,
            "lb": lb, "ub": ub, "x0": x0, "i_eb": i_eb}

    return data


# 使用 setup_test_equal_bounds 函数设置测试数据
eb_data = setup_test_equal_bounds()


# This test is about handling fixed variables, not the accuracy of the solvers
# 标记该测试在 32 位系统上会失败，不是因为逻辑问题而是由于浮点数问题
@pytest.mark.xfail_on_32bit("Failures due to floating point issues, not logic")
@pytest.mark.parametrize('method', eb_data["methods"])
@pytest.mark.parametrize('kwds', eb_data["kwds"])
@pytest.mark.parametrize('bound_type', eb_data["bound_types"])
@pytest.mark.parametrize('constraints', eb_data["constraints"])
@pytest.mark.parametrize('callback', eb_data["callbacks"])
def test_equal_bounds(method, kwds, bound_type, constraints, callback):
    """
    Tests that minimizers still work if (bounds.lb == bounds.ub).any()
    gh12502 - Divide by zero in Jacobian numerical differentiation when
    equality bounds constraints are used
    """
    # GH-15051; slightly more skips than necessary; hopefully fixed by GH-14882
    # 如果运行平台是 'aarch64' 且使用的方法是 "TNC"，且未开启 Jacobian 计算且有回调函数，则跳过测试
    if (platform.machine() == 'aarch64' and method == "TNC"
            and kwds["jac"] is False and callback is not None):
        pytest.skip('Tolerance violation on aarch')

    # 从 eb_data 中获取下列变量
    lb, ub = eb_data["lb"], eb_data["ub"]
    x0, i_eb = eb_data["x0"], eb_data["i_eb"]

    test_constraints, reference_constraints = constraints
    # 如果测试约束存在且方法不是 'SLSQP'，则跳过测试，因为只有 'SLSQP' 支持非线性约束
    if test_constraints and not method == 'SLSQP':
        pytest.skip('Only SLSQP supports nonlinear constraints')

    # 参考约束总是有解析梯度
    # 如果测试约束与参考约束不同，将需要使用有限差分求导
    fd_needed = (test_constraints != reference_constraints)

    # 根据给定的下限 lb 和上限 ub 创建 bounds 对象（旧式或新式）
    bounds = bound_type(lb, ub)

    # 更新 kwds 字典，添加 x0, method, bounds, constraints 和 callback
    kwds.update({"x0": x0, "method": method, "bounds": bounds,
                 "constraints": test_constraints, "callback": callback})
    # 使用 optimize.minimize 函数进行优化
    res = optimize.minimize(**kwds)

    # 使用参考约束运行 optimize.minimize 函数，用于比较输出的解
    expected = optimize.minimize(optimize.rosen, x0, method=method,
                                 jac=optimize.rosen_der, bounds=bounds,
                                 constraints=reference_constraints)

    # 检查结果是否成功
    assert res.success
    # 比较数值解的函数值，设置相对容差 rtol=1.5e-6
    assert_allclose(res.fun, expected.fun, rtol=1.5e-6)
    # 比较数值解的解向量，设置相对容差 rtol=5e-4
    assert_allclose(res.x, expected.x, rtol=5e-4)

    # 如果需要有限差分求导或者用户设置了不使用 Jacobian，将期望的雅可比矩阵中涉及到的位置设为 NaN
    if fd_needed or kwds['jac'] is False:
        expected.jac[i_eb] = np.nan
    # 检查 res 的雅可比矩阵的形状是否为 (4, )
    assert res.jac.shape[0] == 4
    # 比较雅可比矩阵的特定位置值，设置相对容差 rtol=1e-6
    assert_allclose(res.jac[i_eb], expected.jac[i_eb], rtol=1e-6)
    # 如果条件为假，执行以下语句块：kwds['jac'] 为假，test_constraints 为空，bounds 不是 Bounds 类的实例
    # 这段代码用于比较输出与不需要因子分解的等价 FD 最小化的结果
    def fun(x):
        # 创建一个包含 NaN 的新数组，替换其中的特定位置的值为 x 中的值
        new_x = np.array([np.nan, 2, np.nan, -1])
        new_x[[0, 2]] = x
        # 调用 optimize 模块中的 rosen 函数，并传入新数组 new_x 作为参数
        return optimize.rosen(new_x)

    # 使用 optimize 模块中的 minimize 函数，对 fun 函数进行最小化处理
    fd_res = optimize.minimize(fun,
                               x0[[0, 2]],  # 使用 x0 数组的特定位置构成的子数组作为初始点
                               method=method,  # 指定优化方法
                               bounds=bounds[::2])  # 使用 bounds 数组的每隔一个元素构成的子数组作为约束条件

    # 断言 res.fun 与 fd_res.fun 的值在允许的误差范围内相等
    assert_allclose(res.fun, fd_res.fun)

    # TODO 此测试应该与上述因子化版本等效，包括 res.nfev。然而，测试发现当 TNC 被调用时，无论是否有回调，输出都不同。
    # 这两个应该是相同的！这表明当不应该时，TNC 回调可能会改变某些东西。
    assert_allclose(res.x[[0, 2]], fd_res.x, rtol=2e-6)
@pytest.mark.parametrize('method', eb_data["methods"])
def test_all_bounds_equal(method):
    # 这个测试仅适用于那些当 lb==ub 时参数已被拆分的方法
    # 不测试那些与边界一起工作的其他方法

    def f(x, p1=1):
        # 定义一个简单的测试函数 f(x, p1)，计算向量 x 的范数加上参数 p1
        return np.linalg.norm(x) + p1

    bounds = [(1, 1), (2, 2)]
    x0 = (1.0, 3.0)

    # 使用给定的方法调用优化函数 minimize，传入函数 f、初始值 x0 和边界 bounds
    res = optimize.minimize(f, x0, bounds=bounds, method=method)
    assert res.success  # 确保优化成功
    assert_allclose(res.fun, f([1.0, 2.0]))  # 确保优化结果的函数值接近预期值
    assert res.nfev == 1  # 确保只调用了一次函数 f
    assert res.message == 'All independent variables were fixed by bounds.'
    # 确保优化返回消息符合预期

    args = (2,)
    # 以带有参数 args 的方式再次调用 minimize 函数
    res = optimize.minimize(f, x0, bounds=bounds, method=method, args=args)
    assert res.success  # 确保优化成功
    assert_allclose(res.fun, f([1.0, 2.0], 2))  # 确保优化结果的函数值接近预期值

    if method.upper() == 'SLSQP':
        # 对于特定方法 'SLSQP'，定义一个约束函数 con(x)
        def con(x):
            return np.sum(x)

        # 创建一个非线性约束对象 nlc，并调用 minimize 函数进行优化
        nlc = NonlinearConstraint(con, -np.inf, 0.0)
        res = optimize.minimize(
            f, x0, bounds=bounds, method=method, constraints=[nlc]
        )
        assert res.success is False  # 确保优化失败
        assert_allclose(res.fun, f([1.0, 2.0]))  # 确保优化结果的函数值接近预期值
        assert res.nfev == 1  # 确保只调用了一次函数 f
        message = "All independent variables were fixed by bounds, but"
        assert res.message.startswith(message)
        # 确保优化返回消息以特定前缀开头

        # 使用不同的约束上下界创建新的约束对象 nlc，再次调用 minimize 函数进行优化
        nlc = NonlinearConstraint(con, -np.inf, 4)
        res = optimize.minimize(
            f, x0, bounds=bounds, method=method, constraints=[nlc]
        )
        assert res.success is True  # 确保优化成功
        assert_allclose(res.fun, f([1.0, 2.0]))  # 确保优化结果的函数值接近预期值
        assert res.nfev == 1  # 确保只调用了一次函数 f
        message = "All independent variables were fixed by bounds at values"
        assert res.message.startswith(message)
        # 确保优化返回消息以特定前缀开头


def test_eb_constraints():
    # 确保在等边界条件下，约束函数不会被覆盖，参考 GH14859

    def f(x):
        # 定义一个简单的测试函数 f(x)，计算输入向量 x 的多项式
        return x[0]**3 + x[1]**2 + x[2]*x[3]

    def cfun(x):
        # 定义约束函数 cfun(x)，对输入向量 x 的元素求和，返回与常数 40 的差值
        return x[0] + x[1] + x[2] + x[3] - 40

    constraints = [{'type': 'ineq', 'fun': cfun}]

    bounds = [(0, 20)] * 4
    bounds[1] = (5, 5)

    # 使用 SLSQP 方法调用 optimize.minimize 函数，传入函数 f、初始值 x0 和约束条件 constraints
    optimize.minimize(
        f,
        x0=[1, 2, 3, 4],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    # 确保约束列表中的第一个约束函数仍然是 cfun
    assert constraints[0]['fun'] == cfun


def test_show_options():
    solver_methods = {
        'minimize': MINIMIZE_METHODS,
        'minimize_scalar': MINIMIZE_SCALAR_METHODS,
        'root': ROOT_METHODS,
        'root_scalar': ROOT_SCALAR_METHODS,
        'linprog': LINPROG_METHODS,
        'quadratic_assignment': QUADRATIC_ASSIGNMENT_METHODS,
    }

    for solver, methods in solver_methods.items():
        for method in methods:
            # 测试 `show_options` 函数在不出错的情况下运行
            show_options(solver, method)

    unknown_solver_method = {
        'minimize': "ekki",  # 未知方法
        'maximize': "cg",  # 未知求解器
        'maximize_scalar': "ekki",  # 未知求解器和方法
    }
    for solver, method in unknown_solver_method.items():
        # 遍历 unknown_solver_method 字典中的每个键值对，分别赋值给 solver 和 method
        # 断言：测试 show_options 函数调用时是否会引发 ValueError 异常
        assert_raises(ValueError, show_options, solver, method)
def test_bounds_with_list():
    # gh13501. Bounds created with lists weren't working for Powell.
    # 使用列表创建的边界在 Powell 方法中无法正常工作
    bounds = optimize.Bounds(lb=[5., 5.], ub=[10., 10.])
    # 调用 minimize 函数，使用 Powell 方法进行优化，传入定义的边界 bounds
    optimize.minimize(
        optimize.rosen, x0=np.array([9, 9]), method='Powell', bounds=bounds
    )


def test_x_overwritten_user_function():
    # if the user overwrites the x-array in the user function it's likely
    # that the minimizer stops working properly.
    # 如果用户在自定义函数中覆盖了 x 数组，可能会导致优化器无法正常工作
    # gh13740
    def fquad(x):
        # 生成长度与 x 相同的等差数列 a
        a = np.arange(np.size(x))
        # 将 x 减去 a 中的值
        x -= a
        # 计算 x 的平方
        x *= x
        return np.sum(x)

    def fquad_jac(x):
        # 生成长度与 x 相同的等差数列 a
        a = np.arange(np.size(x))
        # 计算梯度向量
        x *= 2
        x -= 2 * a
        return x

    def fquad_hess(x):
        # 返回单位矩阵乘以 2
        return np.eye(np.size(x)) * 2.0

    # 定义需要使用 Jacobian 的优化方法列表
    meth_jac = [
        'newton-cg', 'dogleg', 'trust-ncg', 'trust-exact',
        'trust-krylov', 'trust-constr'
    ]
    # 定义需要使用 Hessian 的优化方法列表
    meth_hess = [
        'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr'
    ]

    # 初始化起始点 x0
    x0 = np.ones(5) * 1.5

    # 对于 MINIMIZE_METHODS 中的每一种方法 meth
    for meth in MINIMIZE_METHODS:
        jac = None
        hess = None
        # 如果 meth 在需要使用 Jacobian 的方法列表中
        if meth in meth_jac:
            jac = fquad_jac
        # 如果 meth 在需要使用 Hessian 的方法列表中
        if meth in meth_hess:
            hess = fquad_hess
        # 调用 minimize 函数进行优化，传入函数 fquad、起始点 x0、方法 meth、以及可能的 Jacobian 和 Hessian
        res = optimize.minimize(fquad, x0, method=meth, jac=jac, hess=hess)
        # 断言优化结果 res 的 x 值接近 x0 的等差数列，允许的误差为 2e-4
        assert_allclose(res.x, np.arange(np.size(x0)), atol=2e-4)


class TestGlobalOptimization:

    def test_optimize_result_attributes(self):
        def func(x):
            return x ** 2

        # Note that `brute` solver does not return `OptimizeResult`
        # 注意 `brute` 求解器不会返回 `OptimizeResult`
        results = [optimize.basinhopping(func, x0=1),
                   optimize.differential_evolution(func, [(-4, 4)]),
                   optimize.shgo(func, [(-4, 4)]),
                   optimize.dual_annealing(func, [(-4, 4)]),
                   optimize.direct(func, [(-4, 4)]),
                   ]

        # 对于每个结果 result
        for result in results:
            # 断言 result 是 OptimizeResult 类型
            assert isinstance(result, optimize.OptimizeResult)
            # 断言 result 具有特定的属性
            assert hasattr(result, "x")
            assert hasattr(result, "success")
            assert hasattr(result, "message")
            assert hasattr(result, "fun")
            assert hasattr(result, "nfev")
            assert hasattr(result, "nit")


def test_approx_fprime():
    # check that approx_fprime (serviced by approx_derivative) works for
    # jac and hess
    # 检查 approx_fprime（由 approx_derivative 提供支持）对 jac 和 hess 是否有效
    g = optimize.approx_fprime(himmelblau_x0, himmelblau)
    # 断言 g 与 himmelblau_grad(himmelblau_x0) 的结果非常接近，相对误差不超过 5e-6
    assert_allclose(g, himmelblau_grad(himmelblau_x0), rtol=5e-6)

    h = optimize.approx_fprime(himmelblau_x0, himmelblau_grad)
    # 断言 h 与 himmelblau_hess(himmelblau_x0) 的结果非常接近，相对误差不超过 5e-6
    assert_allclose(h, himmelblau_hess(himmelblau_x0), rtol=5e-6)


def test_gh12594():
    # gh-12594 reported an error in `_linesearch_powell` and
    # `_line_for_search` when `Bounds` was passed lists instead of arrays.
    # Check that results are the same whether the inputs are lists or arrays.
    # gh-12594 报告了在将 `Bounds` 传递为列表而不是数组时，在 `_linesearch_powell` 和 `_line_for_search` 中出现的错误。
    # 检查输入是列表还是数组时，结果是否相同。

    def f(x):
        return x[0]**2 + (x[1] - 1)**2

    # 创建 Bounds 对象，传入列表形式的下界 lb 和上界 ub
    bounds = Bounds(lb=[-10, -10], ub=[10, 10])
    # 调用 minimize 函数进行优化，传入函数 f、初始点 x0、方法 'Powell' 和边界 bounds
    res = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)
    # 创建包含边界的 Bounds 对象，指定优化变量的下界和上界
    bounds = Bounds(lb=np.array([-10, -10]), ub=np.array([10, 10]))
    
    # 使用 Powell 方法进行优化，通过 minimize 函数调用优化目标函数 f
    # 初始点设定为 (0, 0)，并传入之前创建的边界 bounds
    ref = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)
    
    # 断言优化结果 res 的目标函数值与参考结果 ref 的目标函数值相近
    assert_allclose(res.fun, ref.fun)
    
    # 断言优化结果 res 的最优点（优化变量的值）与参考结果 ref 的最优点相近
    assert_allclose(res.x, ref.x)
# 使用 pytest 的参数化功能，为测试方法参数 `method` 和 `sparse_type` 分别提供不同的取值
@pytest.mark.parametrize('method', ['Newton-CG', 'trust-constr'])
@pytest.mark.parametrize('sparse_type', [coo_matrix, csc_matrix, csr_matrix,
                                         coo_array, csr_array, csc_array])
def test_sparse_hessian(method, sparse_type):
    # 问题 gh-8792 报告了使用 `newton_cg` 进行最小化时，当 `hess` 返回稀疏矩阵时出错。
    # 检查对于接受稀疏 Hessian 矩阵的优化方法，无论 `hess` 返回稠密还是稀疏矩阵，结果应该相同。

    # 定义一个函数 `sparse_rosen_hess`，将 Rosenbrock 函数的 Hessian 矩阵转换为指定的稀疏类型
    def sparse_rosen_hess(x):
        return sparse_type(rosen_hess(x))

    x0 = [2., 2.]

    # 使用 `optimize.minimize` 进行优化，指定方法 `method`，Jacobi 矩阵为 `rosen_der`，Hessian 矩阵为 `sparse_rosen_hess`
    res_sparse = optimize.minimize(rosen, x0, method=method,
                                   jac=rosen_der, hess=sparse_rosen_hess)
    # 再次使用 `optimize.minimize` 进行优化，Hessian 矩阵为 `rosen_hess`，与上述相比为稠密矩阵
    res_dense = optimize.minimize(rosen, x0, method=method,
                                  jac=rosen_der, hess=rosen_hess)

    # 断言稠密矩阵和稀疏矩阵优化结果的目标函数值应接近
    assert_allclose(res_dense.fun, res_sparse.fun)
    # 断言稠密矩阵和稀疏矩阵优化结果的最优点应接近
    assert_allclose(res_dense.x, res_sparse.x)
    # 断言稠密矩阵和稀疏矩阵优化结果的函数评估次数应相等
    assert res_dense.nfev == res_sparse.nfev
    # 断言稠密矩阵和稀疏矩阵优化结果的雅可比矩阵评估次数应相等
    assert res_dense.njev == res_sparse.njev
    # 断言稠密矩阵和稀疏矩阵优化结果的黑塞矩阵评估次数应相等
    assert res_dense.nhev == res_sparse.nhev
```