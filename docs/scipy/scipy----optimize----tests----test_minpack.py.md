# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_minpack.py`

```
"""
Unit tests for optimization routines from minpack.py.
"""
# 导入警告模块和测试框架
import warnings
import pytest

# 导入必要的测试断言和工具函数
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           assert_warns, suppress_warnings)
from pytest import raises as assert_raises

# 导入 NumPy 库及其子模块
import numpy as np
from numpy import array, float64

# 导入多线程池
from multiprocessing.pool import ThreadPool

# 导入 SciPy 库及其子模块
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds


class ReturnShape:
    """This class exists to create a callable that does not have a '__name__' attribute.

    __init__ takes the argument 'shape', which should be a tuple of ints.
    When an instance is called with a single argument 'x', it returns numpy.ones(shape).
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return np.ones(self.shape)


def dummy_func(x, shape):
    """A function that returns an array of ones of the given shape.
    `x` is ignored.
    """
    return np.ones(shape)


def sequence_parallel(fs):
    # 使用线程池并行执行函数列表中的函数
    with ThreadPool(len(fs)) as pool:
        return pool.map(lambda f: f(), fs)


# Function and Jacobian for tests of solvers for systems of nonlinear
# equations


def pressure_network(flow_rates, Qtot, k):
    """Evaluate non-linear equation system representing
    the pressures and flows in a system of n parallel pipes::

        f_i = P_i - P_0, for i = 1..n
        f_0 = sum(Q_i) - Qtot

    where Q_i is the flow rate in pipe i and P_i the pressure in that pipe.
    Pressure is modeled as a P=kQ**2 where k is a valve coefficient and
    Q is the flow rate.

    Parameters
    ----------
    flow_rates : float
        A 1-D array of n flow rates [kg/s].
    k : float
        A 1-D array of n valve coefficients [1/kg m].
    Qtot : float
        A scalar, the total input flow rate [kg/s].

    Returns
    -------
    F : float
        A 1-D array, F[i] == f_i.

    """
    # 计算每个管道的压力
    P = k * flow_rates**2
    # 计算非线性方程组的各个方程
    F = np.hstack((P[1:] - P[0], flow_rates.sum() - Qtot))
    return F


def pressure_network_jacobian(flow_rates, Qtot, k):
    """Return the jacobian of the equation system F(flow_rates)
    computed by `pressure_network` with respect to
    *flow_rates*. See `pressure_network` for the detailed
    description of parameters.

    Returns
    -------
    jac : float
        *n* by *n* matrix ``df_i/dQ_i`` where ``n = len(flow_rates)``
        and *f_i* and *Q_i* are described in the doc for `pressure_network`
    """
    # 确定管道数量
    n = len(flow_rates)
    # 计算每个方程关于流量的偏导数
    pdiff = np.diag(flow_rates[1:] * 2 * k[1:] - 2 * flow_rates[0] * k[0])

    # 初始化雅可比矩阵
    jac = np.empty((n, n))
    jac[:n-1, :n-1] = pdiff * 0
    jac[:n-1, n-1] = 0
    jac[n-1, :] = np.ones(n)

    return jac


def pressure_network_fun_and_grad(flow_rates, Qtot, k):
    # To be continued...
    # 调用 pressure_network 函数和 pressure_network_jacobian 函数，并返回它们的结果作为元组
    return (pressure_network(flow_rates, Qtot, k),
            pressure_network_jacobian(flow_rates, Qtot, k))
    # 定义一个测试类 TestFSolve，用于测试 fsolve 函数的各种情况
    class TestFSolve:
        
        # 测试在没有梯度情况下的 fsolve，等压力管道 -> 等流量
        def test_pressure_network_no_gradient(self):
            # 创建一个包含四个元素，值均为 0.5 的 numpy 数组 k
            k = np.full(4, 0.5)
            # 设定总流量 Qtot 为 4
            Qtot = 4
            # 初始猜测值 initial_guess 为一个包含四个元素的数组 [2., 0., 2., 0.]
            initial_guess = array([2., 0., 2., 0.])
            # 使用 fsolve 求解 pressure_network 函数，返回最终流量、信息、错误码、消息
            final_flows, info, ier, mesg = optimize.fsolve(
                pressure_network, initial_guess, args=(Qtot, k),
                full_output=True)
            # 断言最终流量 final_flows 几乎等于一个包含四个 1 的 numpy 数组
            assert_array_almost_equal(final_flows, np.ones(4))
            # 断言错误码 ier 等于 1，同时检查错误消息 mesg
            assert_(ier == 1, mesg)

        # 测试在有梯度情况下的 fsolve，等压力管道 -> 等流量
        def test_pressure_network_with_gradient(self):
            # 创建一个包含四个元素，值均为 0.5 的 numpy 数组 k
            k = np.full(4, 0.5)
            # 设定总流量 Qtot 为 4
            Qtot = 4
            # 初始猜测值 initial_guess 为一个包含四个元素的数组 [2., 0., 2., 0.]
            initial_guess = array([2., 0., 2., 0.])
            # 使用 fsolve 求解 pressure_network 函数，返回最终流量，使用 pressure_network_jacobian 函数计算梯度
            final_flows = optimize.fsolve(
                pressure_network, initial_guess, args=(Qtot, k),
                fprime=pressure_network_jacobian)
            # 断言最终流量 final_flows 几乎等于一个包含四个 1 的 numpy 数组
            assert_array_almost_equal(final_flows, np.ones(4))

        # 测试函数返回结果与预期不符的情况，预期会抛出 TypeError
        def test_wrong_shape_func_callable(self):
            # 创建一个 ReturnShape(1) 实例 func
            func = ReturnShape(1)
            # 设定初始猜测值 x0 为包含两个元素的列表 [1.5, 2.0]
            x0 = [1.5, 2.0]
            # 断言调用 fsolve 时会抛出 TypeError 异常
            assert_raises(TypeError, optimize.fsolve, func, x0)

        # 测试函数返回结果与预期不符的情况，预期会抛出 TypeError
        def test_wrong_shape_func_function(self):
            # 设定初始猜测值 x0 为包含两个元素的列表 [1.5, 2.0]
            x0 = [1.5, 2.0]
            # 断言调用 fsolve 时会抛出 TypeError 异常，使用 dummy_func 函数作为函数参数
            assert_raises(TypeError, optimize.fsolve, dummy_func, x0, args=((1,),))

        # 测试梯度函数返回结果与预期不符的情况，预期会抛出 TypeError
        def test_wrong_shape_fprime_callable(self):
            # 创建一个 ReturnShape(1) 实例 func
            func = ReturnShape(1)
            # 创建一个 ReturnShape((2,2)) 实例 deriv_func
            deriv_func = ReturnShape((2,2))
            # 断言调用 fsolve 时会抛出 TypeError 异常，使用 deriv_func 作为梯度函数
            assert_raises(TypeError, optimize.fsolve, func, x0=[0,1], fprime=deriv_func)

        # 测试梯度函数返回结果与预期不符的情况，预期会抛出 TypeError
        def test_wrong_shape_fprime_function(self):
            # 定义 func 函数，返回 dummy_func 函数的结果
            def func(x):
                return dummy_func(x, (2,))
            # 定义 deriv_func 函数，返回 dummy_func 函数的结果
            def deriv_func(x):
                return dummy_func(x, (3, 3))
            # 断言调用 fsolve 时会抛出 TypeError 异常，使用 deriv_func 作为梯度函数
            assert_raises(TypeError, optimize.fsolve, func, x0=[0,1], fprime=deriv_func)

        # 测试函数 func 能够抛出 ValueError 异常
        def test_func_can_raise(self):
            # 定义 func 函数，直接抛出 ValueError 异常
            def func(*args):
                raise ValueError('I raised')

            # 使用 assert_raises 断言调用 fsolve 时会抛出 ValueError 异常，并检查错误消息
            with assert_raises(ValueError, match='I raised'):
                optimize.fsolve(func, x0=[0])

        # 测试梯度函数 Dfun 能够抛出 ValueError 异常
        def test_Dfun_can_raise(self):
            # 定义 func 函数，返回 x - [10] 的结果
            def func(x):
                return x - np.array([10])

            # 定义 deriv_func 函数，直接抛出 ValueError 异常
            def deriv_func(*args):
                raise ValueError('I raised')

            # 使用 assert_raises 断言调用 fsolve 时会抛出 ValueError 异常，并检查错误消息
            with assert_raises(ValueError, match='I raised'):
                optimize.fsolve(func, x0=[0], fprime=deriv_func)

        # 测试使用 float32 类型计算的情况
        def test_float32(self):
            # 定义 func 函数，计算 [x[0] - 100, x[1] - 1000] 的平方，返回 float32 类型的数组
            def func(x):
                return np.array([x[0] - 100, x[1] - 1000], dtype=np.float32) ** 2
            # 使用 fsolve 求解 func 函数，初始猜测值为 np.array([1, 1], np.float32)
            p = optimize.fsolve(func, np.array([1, 1], np.float32))
            # 断言使用 fsolve 求解的结果 p，与 [0, 0] 的差值在给定的容差范围内
            assert_allclose(func(p), [0, 0], atol=1e-3)
    # 定义测试函数：测试可重入函数，内部调用无梯度压力网络测试函数并返回压力网络的结果
    def test_reentrant_func(self):
        def func(*args):
            # 调用测试函数：测试无梯度压力网络
            self.test_pressure_network_no_gradient()
            # 返回压力网络函数的结果
            return pressure_network(*args)

        # 创建长度为4的数组 k，每个元素赋值为 0.5
        k = np.full(4, 0.5)
        # 设置总流量 Qtot 为 4
        Qtot = 4
        # 设置初始猜测值为 [2., 0., 2., 0.]
        initial_guess = array([2., 0., 2., 0.])
        # 使用 fsolve 求解压力网络函数，返回最终的流量、信息、错误代码、消息
        final_flows, info, ier, mesg = optimize.fsolve(
            func, initial_guess, args=(Qtot, k),
            full_output=True)
        # 断言最终流量与全为1的数组近似相等
        assert_array_almost_equal(final_flows, np.ones(4))
        # 断言错误代码为1，消息为预期消息
        assert_(ier == 1, mesg)

    # 定义测试函数：测试可重入导数函数，内部调用有梯度压力网络测试函数并返回压力网络的雅可比矩阵
    def test_reentrant_Dfunc(self):
        def deriv_func(*args):
            # 调用测试函数：测试有梯度压力网络
            self.test_pressure_network_with_gradient()
            # 返回压力网络函数的雅可比矩阵
            return pressure_network_jacobian(*args)

        # 创建长度为4的数组 k，每个元素赋值为 0.5
        k = np.full(4, 0.5)
        # 设置总流量 Qtot 为 4
        Qtot = 4
        # 设置初始猜测值为 [2., 0., 2., 0.]
        initial_guess = array([2., 0., 2., 0.])
        # 使用 fsolve 求解压力网络函数，传入雅可比矩阵函数
        final_flows = optimize.fsolve(
            pressure_network, initial_guess, args=(Qtot, k),
            fprime=deriv_func)
        # 断言最终流量与全为1的数组近似相等
        assert_array_almost_equal(final_flows, np.ones(4))

    # 定义测试函数：测试无梯度并行执行
    def test_concurrent_no_gradient(self):
        # 并行执行测试函数：测试无梯度压力网络，执行10次
        v = sequence_parallel([self.test_pressure_network_no_gradient] * 10)
        # 断言所有结果为 None
        assert all([result is None for result in v])

    # 定义测试函数：测试有梯度并行执行
    def test_concurrent_with_gradient(self):
        # 并行执行测试函数：测试有梯度压力网络，执行10次
        v = sequence_parallel([self.test_pressure_network_with_gradient] * 10)
        # 断言所有结果为 None
        assert all([result is None for result in v])
class TestRootHybr:
    def test_pressure_network_no_gradient(self):
        # root/hybr without gradient, equal pipes -> equal flows
        # 定义管道阻力系数数组，每个管道阻力为0.5
        k = np.full(4, 0.5)
        # 总流量设定为4
        Qtot = 4
        # 初始猜测解数组，设定为[2., 0., 2., 0.]
        initial_guess = array([2., 0., 2., 0.])
        # 使用混合方法（hybr）优化求解流量分布，并返回最终的流量数组
        final_flows = optimize.root(pressure_network, initial_guess,
                                    method='hybr', args=(Qtot, k)).x
        # 断言最终的流量数组接近全为1的数组
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_pressure_network_with_gradient(self):
        # root/hybr with gradient, equal pipes -> equal flows
        # 定义管道阻力系数数组，每个管道阻力为0.5
        k = np.full(4, 0.5)
        # 总流量设定为4
        Qtot = 4
        # 初始猜测解数组，设定为[[2., 0., 2., 0.]]
        initial_guess = array([[2., 0., 2., 0.]])
        # 使用混合方法（hybr）优化求解流量分布，并返回最终的流量数组
        final_flows = optimize.root(pressure_network, initial_guess,
                                    args=(Qtot, k), method='hybr',
                                    jac=pressure_network_jacobian).x
        # 断言最终的流量数组接近全为1的数组
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_pressure_network_with_gradient_combined(self):
        # root/hybr with gradient and function combined, equal pipes -> equal
        # flows
        # 定义管道阻力系数数组，每个管道阻力为0.5
        k = np.full(4, 0.5)
        # 总流量设定为4
        Qtot = 4
        # 初始猜测解数组，设定为[2., 0., 2., 0.]
        initial_guess = array([2., 0., 2., 0.])
        # 使用混合方法（hybr）优化求解流量分布，并返回最终的流量数组
        final_flows = optimize.root(pressure_network_fun_and_grad,
                                    initial_guess, args=(Qtot, k),
                                    method='hybr', jac=True).x
        # 断言最终的流量数组接近全为1的数组
        assert_array_almost_equal(final_flows, np.ones(4))


class TestRootLM:
    def test_pressure_network_no_gradient(self):
        # root/lm without gradient, equal pipes -> equal flows
        # 定义管道阻力系数数组，每个管道阻力为0.5
        k = np.full(4, 0.5)
        # 总流量设定为4
        Qtot = 4
        # 初始猜测解数组，设定为[2., 0., 2., 0.]
        initial_guess = array([2., 0., 2., 0.])
        # 使用Levenberg-Marquardt方法（lm）优化求解流量分布，并返回最终的流量数组
        final_flows = optimize.root(pressure_network, initial_guess,
                                    method='lm', args=(Qtot, k)).x
        # 断言最终的流量数组接近全为1的数组
        assert_array_almost_equal(final_flows, np.ones(4))


class TestNfev:
    def zero_f(self, y):
        # 计数器自增，用于计算函数调用次数
        self.nfev += 1
        # 返回计算结果，y^2-3
        return y**2-3

    @pytest.mark.parametrize('method', ['hybr', 'lm', 'broyden1',
                                        'broyden2', 'anderson',
                                        'linearmixing', 'diagbroyden',
                                        'excitingmixing', 'krylov',
                                        'df-sane'])
    def test_root_nfev(self, method):
        # 初始化计数器
        self.nfev = 0
        # 使用不同的优化方法求解零点，断言计算函数调用次数与预期相同
        solution = optimize.root(self.zero_f, 100, method=method)
        assert solution.nfev == self.nfev

    def test_fsolve_nfev(self):
        # 初始化计数器
        self.nfev = 0
        # 使用fsolve函数求解零点，断言计算函数调用次数与预期相同
        x, info, ier, mesg = optimize.fsolve(self.zero_f, 100, full_output=True)
        assert info['nfev'] == self.nfev


class TestLeastSq:
    def setup_method(self):
        # 生成数据点x
        x = np.linspace(0, 10, 40)
        # 设置参数a, b, c
        a,b,c = 3.1, 42, -304.2
        self.x = x
        self.abc = a,b,c
        # 生成真实数据y_true，并添加随机误差
        y_true = a*x**2 + b*x + c
        np.random.seed(0)
        self.y_meas = y_true + 0.01*np.random.standard_normal(y_true.shape)

    def residuals(self, p, y, x):
        # 定义残差函数，返回观测值与拟合值之间的误差
        a,b,c = p
        err = y-(a*x**2 + b*x + c)
        return err
    # 定义一个方法用于计算残差的雅可比矩阵，对给定的参数 `_p`、观测值 `_y` 和自变量 `x` 进行计算
    def residuals_jacobian(self, _p, _y, x):
        # 返回的结果是 x**2、x 和 x 全为 1 的矩阵的负数垂直堆叠
        return -np.vstack([x**2, x, np.ones_like(x)]).T

    # 进行基本的测试
    def test_basic(self):
        # 初始参数 p0 是一个数组 [0,0,0]
        p0 = array([0,0,0])
        # 使用最小二乘法拟合残差函数 `self.residuals`，传入参数 p0、观测值 `self.y_meas` 和自变量 `self.x`
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x))
        # 确保返回的 ier 在 (1,2,3,4) 中，否则输出错误信息
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # 由于随机性较低，使用 assert_array_almost_equal 近似比较 params_fit 和 self.abc，精度为 2
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    # 带有梯度函数的基本测试
    def test_basic_with_gradient(self):
        # 初始参数 p0 是一个数组 [0,0,0]
        p0 = array([0,0,0])
        # 使用最小二乘法拟合残差函数 `self.residuals`，传入参数 p0、观测值 `self.y_meas` 和自变量 `self.x`，以及梯度函数 `self.residuals_jacobian`
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x),
                                  Dfun=self.residuals_jacobian)
        # 确保返回的 ier 在 (1,2,3,4) 中，否则输出错误信息
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # 由于随机性较低，使用 assert_array_almost_equal 近似比较 params_fit 和 self.abc，精度为 2
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    # 测试完整输出
    def test_full_output(self):
        # 初始参数 p0 是一个形状为 (1,3) 的数组 [[0,0,0]]
        p0 = array([[0,0,0]])
        # 使用最小二乘法拟合残差函数 `self.residuals`，传入参数 p0、观测值 `self.y_meas` 和自变量 `self.x`，并请求完整输出
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        # 解析完整输出的各个部分：params_fit 拟合后的参数，cov_x 参数估计的协方差矩阵，infodict 优化过程信息，mesg 最终的状态信息，ier 操作状态码
        params_fit, cov_x, infodict, mesg, ier = full_output
        # 确保返回的 ier 在 (1,2,3,4) 中，否则输出自定义的错误信息
        assert_(ier in (1,2,3,4), f'solution not found: {mesg}')

    # 测试输入保持不变
    def test_input_untouched(self):
        # 初始参数 p0 是一个浮点型数组 [0,0,0]
        p0 = array([0,0,0],dtype=float64)
        # 复制一份 p0 的副本 p0_copy
        p0_copy = array(p0, copy=True)
        # 使用最小二乘法拟合残差函数 `self.residuals`，传入参数 p0、观测值 `self.y_meas` 和自变量 `self.x`，并请求完整输出
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        # 解析完整输出的各个部分：params_fit 拟合后的参数，cov_x 参数估计的协方差矩阵，infodict 优化过程信息，mesg 最终的状态信息，ier 操作状态码
        params_fit, cov_x, infodict, mesg, ier = full_output
        # 确保返回的 ier 在 (1,2,3,4) 中，否则输出自定义的错误信息
        assert_(ier in (1,2,3,4), f'solution not found: {mesg}')
        # 确保 p0 和 p0_copy 相等，否则抛出错误
        assert_array_equal(p0, p0_copy)

    # 测试错误的函数返回形状（可调用对象）
    def test_wrong_shape_func_callable(self):
        # 创建一个返回长度为 1 的形状对象 ReturnShape(1)
        func = ReturnShape(1)
        # 设置 x0 为一个包含两个元素的列表，但 func 将返回长度为 1 的数组，因此应该引发 TypeError
        x0 = [1.5, 2.0]
        # 确保调用 optimize.leastsq(func, x0) 会引发 TypeError
        assert_raises(TypeError, optimize.leastsq, func, x0)

    # 测试错误的函数返回形状（函数）
    def test_wrong_shape_func_function(self):
        # 设置 x0 为一个包含两个元素的列表，但 dummy_func 将返回长度为 1 的数组，因此应该引发 TypeError
        x0 = [1.5, 2.0]
        # 确保调用 optimize.leastsq(dummy_func, x0, args=((1,),)) 会引发 TypeError
        assert_raises(TypeError, optimize.leastsq, dummy_func, x0, args=((1,),))

    # 测试错误的雅可比矩阵返回形状（可调用对象）
    def test_wrong_shape_Dfun_callable(self):
        # 创建一个返回形状为 (1,1) 的形状对象 ReturnShape(1)
        func = ReturnShape(1)
        # 创建一个返回形状为 (2,2) 的导数函数对象 ReturnShape((2,2))
        deriv_func = ReturnShape((2,2))
        # 确保调用 optimize.leastsq(func, x0=[0,1], Dfun=deriv_func) 会引发 TypeError
        assert_raises(TypeError, optimize.leastsq, func, x0=[0,1], Dfun=deriv_func)

    # 测试错误的雅可比矩阵返回形状（函数）
    def test_wrong_shape_Dfun_function(self):
        # 定义一个函数 func 返回 dummy_func(x, (2,))
        def func(x):
            return dummy_func(x, (2,))
        # 定义一个函数 deriv_func 返回 dummy_func(x, (3,3))
        def deriv_func(x):
            return dummy_func(x, (3, 3))
        # 确保调用 optimize.leastsq(func, x0=[0,1], Dfun=deriv_func) 会引发 TypeError
        assert_raises(TypeError, optimize.leastsq, func, x0=[0,1], Dfun=deriv_func)
    def test_float32(self):
        # 回归测试 gh-1447，验证 float32 类型的优化
        def func(p,x,y):
            # 定义函数 q，用于拟合数据
            q = p[0]*np.exp(-(x-p[1])**2/(2.0*p[2]**2))+p[3]
            return q - y

        # 定义输入数据 x 和对应的目标 y，都是 float32 类型的数组
        x = np.array([1.475,1.429,1.409,1.419,1.455,1.519,1.472, 1.368,1.286,
                       1.231], dtype=np.float32)
        y = np.array([0.0168,0.0193,0.0211,0.0202,0.0171,0.0151,0.0185,0.0258,
                      0.034,0.0396], dtype=np.float32)
        # 初始参数 p0
        p0 = np.array([1.0,1.0,1.0,1.0])
        # 使用最小二乘法优化参数 p1
        p1, success = optimize.leastsq(func, p0, args=(x,y))

        # 验证优化成功的状态码
        assert_(success in [1,2,3,4])
        # 验证优化后的函数值的平方和与初始值的平方和的比较
        assert_((func(p1,x,y)**2).sum() < 1e-4 * (func(p0,x,y)**2).sum())

    def test_func_can_raise(self):
        # 测试函数能否抛出异常
        def func(*args):
            raise ValueError('I raised')

        # 使用 assert_raises 验证函数是否抛出指定异常和消息
        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0])

    def test_Dfun_can_raise(self):
        # 测试导数函数是否能抛出异常
        def func(x):
            return x - np.array([10])

        def deriv_func(*args):
            raise ValueError('I raised')

        # 使用 assert_raises 验证导数函数是否抛出指定异常和消息
        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0], Dfun=deriv_func)

    def test_reentrant_func(self):
        # 测试可重入函数功能
        def func(*args):
            # 调用 test_basic 方法
            self.test_basic()
            # 返回残差函数的结果
            return self.residuals(*args)

        # 初始参数 p0
        p0 = array([0,0,0])
        # 使用最小二乘法优化参数 params_fit
        params_fit, ier = leastsq(func, p0,
                                  args=(self.y_meas, self.x))
        # 验证优化结果状态码
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # 由于随机性，使用低精度验证 params_fit 和 self.abc 的近似程度
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_reentrant_Dfun(self):
        # 测试可重入导数函数功能
        def deriv_func(*args):
            # 调用 test_basic 方法
            self.test_basic()
            # 返回残差雅可比矩阵的结果
            return self.residuals_jacobian(*args)

        # 初始参数 p0
        p0 = array([0,0,0])
        # 使用最小二乘法优化参数 params_fit
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x),
                                  Dfun=deriv_func)
        # 验证优化结果状态码
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # 由于随机性，使用低精度验证 params_fit 和 self.abc 的近似程度
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_concurrent_no_gradient(self):
        # 测试无梯度的并发功能
        v = sequence_parallel([self.test_basic] * 10)
        # 验证并发执行结果是否全部为 None
        assert all([result is None for result in v])

    def test_concurrent_with_gradient(self):
        # 测试带梯度的并发功能
        v = sequence_parallel([self.test_basic_with_gradient] * 10)
        # 验证并发执行结果是否全部为 None
        assert all([result is None for result in v])

    def test_func_input_output_length_check(self):
        # 测试函数输入输出长度检查
        def func(x):
            # 定义简单的二次函数
            return 2 * (x[0] - 3) ** 2 + 1

        # 使用 assert_raises 验证是否抛出指定的 TypeError 异常和消息
        with assert_raises(TypeError,
                           match='Improper input: func input vector length N='):
            optimize.leastsq(func, x0=[0, 1])
class TestCurveFit:
    # 定义测试类 TestCurveFit
    def setup_method(self):
        # 设置每个测试方法的前置条件：定义测试数据 y 和 x
        self.y = array([1.0, 3.2, 9.5, 13.7])
        self.x = array([1.0, 2.0, 3.0, 4.0])

    # 定义测试方法：测试函数接受一个参数的情况
    def test_one_argument(self):
        # 定义一个接受一个参数的函数 func(x,a)，用于拟合曲线
        def func(x,a):
            return x**a
        # 使用 curve_fit 函数拟合曲线，返回参数 popt 和协方差 pcov
        popt, pcov = curve_fit(func, self.x, self.y)
        # 断言拟合参数 popt 的长度为 1
        assert_(len(popt) == 1)
        # 断言协方差矩阵 pcov 的形状为 (1,1)
        assert_(pcov.shape == (1,1))
        # 断言拟合参数 popt[0] 的值接近于 1.9149，精度为小数点后四位
        assert_almost_equal(popt[0], 1.9149, decimal=4)
        # 断言协方差矩阵 pcov[0,0] 的值接近于 0.0016，精度为小数点后四位
        assert_almost_equal(pcov[0,0], 0.0016, decimal=4)

        # 测试 full_output 参数是否能够得到相同的结果。用于回归测试 issue #1415
        # 同时测试 check_finite 是否可以被关闭
        res = curve_fit(func, self.x, self.y,
                        full_output=1, check_finite=False)
        # 解包 res 得到拟合参数 popt2、协方差 pcov2、信息字典 infodict 等
        (popt2, pcov2, infodict, errmsg, ier) = res
        # 断言拟合参数 popt 与 popt2 的值近似相等
        assert_array_almost_equal(popt, popt2)

    # 定义测试方法：测试函数接受两个参数的情况
    def test_two_argument(self):
        # 定义一个接受两个参数的函数 func(x,a,b)，用于拟合曲线
        def func(x, a, b):
            return b*x**a
        # 使用 curve_fit 函数拟合曲线，返回参数 popt 和协方差 pcov
        popt, pcov = curve_fit(func, self.x, self.y)
        # 断言拟合参数 popt 的长度为 2
        assert_(len(popt) == 2)
        # 断言协方差矩阵 pcov 的形状为 (2,2)
        assert_(pcov.shape == (2,2))
        # 断言拟合参数 popt 的值接近于 [1.7989, 1.1642]，精度为小数点后四位
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        # 断言协方差矩阵 pcov 的值接近于 [[0.0852, -0.1260], [-0.1260, 0.1912]]，精度为小数点后四位
        assert_array_almost_equal(pcov, [[0.0852, -0.1260], [-0.1260, 0.1912]],
                                  decimal=4)

    # 定义测试方法：测试函数作为类方法时的情况
    def test_func_is_classmethod(self):
        # 定义一个类 test_self，用于测试 curve_fit 在模型函数是类实例方法时传递正确数量的参数
        class test_self:
            """This class tests if curve_fit passes the correct number of
               arguments when the model function is a class instance method.
            """

            # 定义一个类实例方法 func(self, x, a, b)，用于拟合曲线
            def func(self, x, a, b):
                return b * x**a

        # 创建 test_self 的实例 test_self_inst
        test_self_inst = test_self()
        # 使用 curve_fit 函数拟合曲线，返回参数 popt 和协方差 pcov
        popt, pcov = curve_fit(test_self_inst.func, self.x, self.y)
        # 断言协方差矩阵 pcov 的形状为 (2,2)
        assert_(pcov.shape == (2,2))
        # 断言拟合参数 popt 的值接近于 [1.7989, 1.1642]，精度为小数点后四位
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        # 断言协方差矩阵 pcov 的值接近于 [[0.0852, -0.1260], [-0.1260, 0.1912]]，精度为小数点后四位
        assert_array_almost_equal(pcov, [[0.0852, -0.1260], [-0.1260, 0.1912]],
                                  decimal=4)

    # 定义测试方法：测试回归问题 #2639
    def test_regression_2639(self):
        # 这个测试在 leastsq 中 epsfcn 过大时会失败
        x = [574.14200000000005, 574.154, 574.16499999999996,
             574.17700000000002, 574.18799999999999, 574.19899999999996,
             574.21100000000001, 574.22199999999998, 574.23400000000004,
             574.245]
        y = [859.0, 997.0, 1699.0, 2604.0, 2013.0, 1964.0, 2435.0,
             1550.0, 949.0, 841.0]
        guess = [574.1861428571428, 574.2155714285715, 1302.0, 1302.0,
                 0.0035019999999983615, 859.0]
        good = [5.74177150e+02, 5.74209188e+02, 1.74187044e+03, 1.58646166e+03,
                1.0068462e-02, 8.57450661e+02]

        # 定义一个双高斯函数 f_double_gauss(x, x0, x1, A0, A1, sigma, c)，用于拟合曲线
        def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
            return (A0*np.exp(-(x-x0)**2/(2.*sigma**2))
                    + A1*np.exp(-(x-x1)**2/(2.*sigma**2)) + c)
        # 使用 curve_fit 函数拟合曲线，返回参数 popt 和协方差 pcov
        popt, pcov = curve_fit(f_double_gauss, x, y, guess, maxfev=10000)
        # 断言拟合参数 popt 与 good 的值接近
        assert_allclose(popt, good, rtol=1e-5)
    def test_pcov(self):
        # 定义测试数据
        xdata = np.array([0, 1, 2, 3, 4, 5])
        ydata = np.array([1, 1, 5, 7, 8, 12])
        sigma = np.array([1, 2, 1, 2, 1, 2])

        # 定义拟合函数
        def f(x, a, b):
            return a*x + b

        # 遍历不同的拟合方法
        for method in ['lm', 'trf', 'dogbox']:
            # 进行曲线拟合，计算参数和协方差矩阵
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma,
                                   method=method)
            # 计算标准误差
            perr_scaled = np.sqrt(np.diag(pcov))
            # 断言标准误差的值在预期范围内
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=1e-3)

            # 使用放大后的标准差进行曲线拟合
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3*sigma,
                                   method=method)
            # 计算放大后的标准误差
            perr_scaled = np.sqrt(np.diag(pcov))
            # 断言放大后的标准误差的值在预期范围内
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=1e-3)

            # 使用绝对标准差进行曲线拟合
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma,
                                   absolute_sigma=True, method=method)
            # 计算绝对标准误差
            perr = np.sqrt(np.diag(pcov))
            # 断言绝对标准误差的值在预期范围内
            assert_allclose(perr, [0.30714756, 0.85045308], rtol=1e-3)

            # 使用放大后的绝对标准差进行曲线拟合
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3*sigma,
                                   absolute_sigma=True, method=method)
            # 计算放大后的绝对标准误差
            perr = np.sqrt(np.diag(pcov))
            # 断言放大后的绝对标准误差的值在预期范围内
            assert_allclose(perr, [3*0.30714756, 3*0.85045308], rtol=1e-3)

        # 无限方差情况

        # 定义平坦函数
        def f_flat(x, a, b):
            return a*x

        # 预期的协方差矩阵全为无穷
        pcov_expected = np.array([np.inf]*4).reshape(2, 2)

        # 忽略警告，进行曲线拟合
        with suppress_warnings() as sup:
            # 设置警告过滤条件
            sup.filter(OptimizeWarning,
                       "Covariance of the parameters could not be estimated")
            # 对平坦函数进行曲线拟合
            popt, pcov = curve_fit(f_flat, xdata, ydata, p0=[2, 0], sigma=sigma)
            # 对部分数据进行曲线拟合
            popt1, pcov1 = curve_fit(f, xdata[:2], ydata[:2], p0=[2, 0])

        # 断言协方差矩阵的形状符合预期
        assert_(pcov.shape == (2, 2))
        # 断言协方差矩阵与预期值相等
        assert_array_equal(pcov, pcov_expected)

        # 断言协方差矩阵的形状符合预期
        assert_(pcov1.shape == (2, 2))
        # 断言协方差矩阵与预期值相等
        assert_array_equal(pcov1, pcov_expected)

    def test_array_like(self):
        # 测试序列输入，用于回归测试 gh-3037
        def f_linear(x, a, b):
            return a*x + b

        # 定义输入序列
        x = [1, 2, 3, 4]
        y = [3, 5, 7, 9]
        # 断言曲线拟合结果与预期值接近
        assert_allclose(curve_fit(f_linear, x, y)[0], [2, 1], atol=1e-10)

    def test_indeterminate_covariance(self):
        # 测试当协方差不确定时是否返回警告
        xdata = np.array([1, 2, 3, 4, 5, 6])
        ydata = np.array([1, 2, 3, 4, 5.5, 6])
        # 断言是否出现优化警告
        assert_warns(OptimizeWarning, curve_fit,
                     lambda x, a, b: a*x, xdata, ydata)
    def test_NaN_handling(self):
        # Test for correct handling of NaNs in input data: gh-3422

        # create input with NaNs
        xdata = np.array([1, np.nan, 3])  # 创建包含 NaN 的输入数组
        ydata = np.array([1, 2, 3])       # 创建正常的输入数组

        # 测试对包含 NaN 的 xdata 和 ydata 调用 curve_fit 是否会引发 ValueError
        assert_raises(ValueError, curve_fit,
                      lambda x, a, b: a*x + b, xdata, ydata)
        assert_raises(ValueError, curve_fit,
                      lambda x, a, b: a*x + b, ydata, xdata)

        # 测试在 check_finite=True 的情况下，对包含 NaN 的输入调用 curve_fit 是否会引发 ValueError
        assert_raises(ValueError, curve_fit, lambda x, a, b: a*x + b,
                      xdata, ydata, **{"check_finite": True})

    @staticmethod
    def _check_nan_policy(f, xdata_with_nan, xdata_without_nan,
                          ydata_with_nan, ydata_without_nan, method):
        kwargs = {'f': f, 'xdata': xdata_with_nan, 'ydata': ydata_with_nan,
                  'method': method, 'check_finite': False}
        # propagate test
        error_msg = ("`nan_policy='propagate'` is not supported "
                     "by this function.")
        # 测试当 nan_policy 设置为 'propagate' 时是否会引发 ValueError
        with assert_raises(ValueError, match=error_msg):
            curve_fit(**kwargs, nan_policy="propagate", maxfev=2000)

        # raise test
        # 测试当 nan_policy 设置为 'raise' 时是否会引发 ValueError
        with assert_raises(ValueError, match="The input contains nan"):
            curve_fit(**kwargs, nan_policy="raise")

        # omit test
        # 测试当 nan_policy 设置为 'omit' 时是否能正常运行，并与没有 NaN 的情况下的结果进行比较
        result_with_nan, _ = curve_fit(**kwargs, nan_policy="omit")
        kwargs['xdata'] = xdata_without_nan
        kwargs['ydata'] = ydata_without_nan
        result_without_nan, _ = curve_fit(**kwargs)
        assert_allclose(result_with_nan, result_without_nan)

        # not valid policy test
        # 测试当 nan_policy 设置为不支持的值时是否会引发 ValueError，并匹配特定的错误消息
        # 这里检查是否能接受参数名称的任意顺序
        error_msg = (r"nan_policy must be one of \{(?:'raise'|'omit'|None)"
                     r"(?:, ?(?:'raise'|'omit'|None))*\}")
        with assert_raises(ValueError, match=error_msg):
            curve_fit(**kwargs, nan_policy="hi")

    @pytest.mark.parametrize('method', ["lm", "trf", "dogbox"])
    def test_nan_policy_1d(self, method):
        def f(x, a, b):
            return a*x + b

        xdata_with_nan = np.array([2, 3, np.nan, 4, 4, np.nan])  # 创建包含 NaN 的输入数组
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7])       # 创建包含 NaN 的输入数组
        xdata_without_nan = np.array([2, 3, 4])                  # 创建不含 NaN 的输入数组
        ydata_without_nan = np.array([1, 2, 3])                  # 创建不含 NaN 的输入数组

        # 调用 _check_nan_policy 方法，测试不同的 nan_policy 对 curve_fit 的影响
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan,
                               ydata_with_nan, ydata_without_nan, method)

    @pytest.mark.parametrize('method', ["lm", "trf", "dogbox"])
    def test_nan_policy_2d(self, method):
        # 定义一个二维函数 f，其中 x 是一个二维数组，a 和 b 是参数
        def f(x, a, b):
            # 提取 x 的第一行和第二行作为 x1 和 x2
            x1 = x[0, :]
            x2 = x[1, :]
            # 返回 a*x1 + b + x2 的计算结果
            return a*x1 + b + x2

        # 包含 NaN 值的 x 数据
        xdata_with_nan = np.array([[2, 3, np.nan, 4, 4, np.nan, 5],
                                   [2, 3, np.nan, np.nan, 4, np.nan, 7]])
        # 包含 NaN 值的 y 数据
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7, 10])
        # 不包含 NaN 值的 x 数据
        xdata_without_nan = np.array([[2, 3, 5], [2, 3, 7]])
        # 不包含 NaN 值的 y 数据
        ydata_without_nan = np.array([1, 2, 10])

        # 调用 self._check_nan_policy 方法，验证 NaN 策略
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan,
                               ydata_with_nan, ydata_without_nan, method)

    @pytest.mark.parametrize('n', [2, 3])
    @pytest.mark.parametrize('method', ["lm", "trf", "dogbox"])
    def test_nan_policy_2_3d(self, n, method):
        # 定义一个二到三维的函数 f，其中 x 是一个二到三维数组，a 和 b 是参数
        def f(x, a, b):
            # 从 x 中提取第一个和第二个轴上的数据，去除单维度后，得到 x1 和 x2
            x1 = x[..., 0, :].squeeze()
            x2 = x[..., 1, :].squeeze()
            # 返回 a*x1 + b + x2 的计算结果
            return a*x1 + b + x2

        # 包含 NaN 值的 x 数据
        xdata_with_nan = np.array([[[2, 3, np.nan, 4, 4, np.nan, 5],
                                   [2, 3, np.nan, np.nan, 4, np.nan, 7]]])
        # 如果 n 等于 2，则将 xdata_with_nan 去除单维度
        xdata_with_nan = xdata_with_nan.squeeze() if n == 2 else xdata_with_nan
        # 包含 NaN 值的 y 数据
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7, 10])
        # 不包含 NaN 值的 x 数据
        xdata_without_nan = np.array([[[2, 3, 5], [2, 3, 7]]])
        # 不包含 NaN 值的 y 数据
        ydata_without_nan = np.array([1, 2, 10])

        # 调用 self._check_nan_policy 方法，验证 NaN 策略
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan,
                               ydata_with_nan, ydata_without_nan, method)

    def test_empty_inputs(self):
        # 测试空输入时是否引发 ValueError 异常，包括带界限和不带界限的情况
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [], [],
                      bounds=(1, 2))
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [1], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [2], [],
                      bounds=(1, 2))

    def test_function_zero_params(self):
        # 测试参数数量为零时是否引发 ValueError 异常
        assert_raises(ValueError, curve_fit, lambda x: x, [1, 2], [3, 4])

    def test_None_x(self):  # Added in GH10196
        # 测试 x 参数为 None 时是否能够正确拟合曲线
        popt, pcov = curve_fit(lambda _, a: a * np.arange(10),
                               None, 2 * np.arange(10))
        # 断言拟合结果 popt 是否接近 [2.]
        assert_allclose(popt, [2.])

    def test_method_argument(self):
        # 定义一个带方法参数的函数 f
        def f(x, a, b):
            return a * np.exp(-b*x)

        # 生成测试数据 xdata 和 ydata
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # 对于不同的方法进行循环测试
        for method in ['trf', 'dogbox', 'lm', None]:
            # 调用 curve_fit 方法进行拟合
            popt, pcov = curve_fit(f, xdata, ydata, method=method)
            # 断言拟合结果 popt 是否接近 [2., 2.]
            assert_allclose(popt, [2., 2.])

        # 测试使用未知方法是否引发 ValueError 异常
        assert_raises(ValueError, curve_fit, f, xdata, ydata, method='unknown')
    def test_full_output(self):
        # 定义一个简单的指数函数
        def f(x, a, b):
            return a * np.exp(-b * x)

        # 生成一组输入数据和对应的输出数据
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # 使用不同的方法拟合曲线，获取详细输出
        for method in ['trf', 'dogbox', 'lm', None]:
            # 调用 curve_fit 进行拟合，获取拟合参数、协方差、详细信息、错误消息和状态码
            popt, pcov, infodict, errmsg, ier = curve_fit(
                f, xdata, ydata, method=method, full_output=True)
            # 检查拟合参数是否准确
            assert_allclose(popt, [2., 2.])
            # 检查 infodict 中是否包含必要的信息
            assert "nfev" in infodict
            assert "fvec" in infodict
            if method == 'lm' or method is None:
                # 对于 'lm' 方法或无方法的情况，需要更详细的信息
                assert "fjac" in infodict
                assert "ipvt" in infodict
                assert "qtf" in infodict
            # 错误消息应为字符串类型
            assert isinstance(errmsg, str)
            # 状态码应在特定范围内
            assert ier in (1, 2, 3, 4)

    def test_bounds(self):
        # 定义一个简单的指数函数
        def f(x, a, b):
            return a * np.exp(-b*x)

        # 生成一组输入数据和对应的输出数据
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # 定义参数的下界和上界
        lb = [1., 0]
        ub = [1.5, 3.]

        # 创建包含下界和上界的 Bounds 对象
        bounds = (lb, ub)
        bounds_class = Bounds(lb, ub)

        # 使用不同的方法和不同的 Bounds 进行拟合，比较结果
        for method in [None, 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, bounds=bounds,
                                   method=method)
            assert_allclose(popt[0], 1.5)

            popt_class, pcov_class = curve_fit(f, xdata, ydata,
                                               bounds=bounds_class,
                                               method=method)
            assert_allclose(popt_class, popt)

        # 对于带有 Bounds 的情况，起始估计应该是可行的
        popt, pcov = curve_fit(f, xdata, ydata, method='trf',
                               bounds=([0., 0], [0.6, np.inf]))
        assert_allclose(popt[0], 0.6)

        # 'lm' 方法不支持使用 Bounds
        assert_raises(ValueError, curve_fit, f, xdata, ydata, bounds=bounds,
                      method='lm')

    def test_bounds_p0(self):
        # 这个测试是针对问题 #5719 的。问题是 'trf' 或 'dogbox' 方法在调用时忽略了初始猜测。
        def f(x, a):
            return np.sin(x + a)

        # 生成一组输入数据和对应的输出数据
        xdata = np.linspace(-2*np.pi, 2*np.pi, 40)
        ydata = np.sin(xdata)

        # 定义初始猜测和边界
        bounds = (-3 * np.pi, 3 * np.pi)

        # 对 'trf' 和 'dogbox' 方法分别进行测试
        for method in ['trf', 'dogbox']:
            # 测试在指定初始猜测和边界的情况下是否得到一致的结果
            popt_1, _ = curve_fit(f, xdata, ydata, p0=2.1*np.pi)
            popt_2, _ = curve_fit(f, xdata, ydata, p0=2.1*np.pi,
                                  bounds=bounds, method=method)

            # 如果初始猜测被忽略，popt_2 应该接近 0
            assert_allclose(popt_1, popt_2)
    def test_jac(self):
        # 测试 Jacobian 可调用函数是否正确处理，并在提供 sigma 时加权。
        def f(x, a, b):
            return a * np.exp(-b*x)

        def jac(x, a, b):
            e = np.exp(-b*x)
            return np.vstack((e, -a * x * e)).T

        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # 测试 least_squares 后端的数值选项。
        for method in ['trf', 'dogbox']:
            for scheme in ['2-point', '3-point', 'cs']:
                popt, pcov = curve_fit(f, xdata, ydata, jac=scheme,
                                       method=method)
                assert_allclose(popt, [2, 2])

        # 测试解析选项。
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, method=method, jac=jac)
            assert_allclose(popt, [2, 2])

        # 现在添加一个异常值并提供 sigma。
        ydata[5] = 100
        sigma = np.ones(xdata.shape[0])
        sigma[5] = 200
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, sigma=sigma, method=method,
                                   jac=jac)
            # 优化过程仍然会受到影响，必须设置 rtol=1e-3。
            assert_allclose(popt, [2, 2], rtol=1e-3)

    def test_maxfev_and_bounds(self):
        # gh-6340: 在没有边界条件时，curve_fit 可接受参数 maxfev（通过 leastsq），
        # 但是有边界条件时，参数是 max_nfev（通过 least_squares）。
        x = np.arange(0, 10)
        y = 2*x
        popt1, _ = curve_fit(lambda x,p: p*x, x, y, bounds=(0, 3), maxfev=100)
        popt2, _ = curve_fit(lambda x,p: p*x, x, y, bounds=(0, 3), max_nfev=100)

        assert_allclose(popt1, 2, atol=1e-14)
        assert_allclose(popt2, 2, atol=1e-14)

    def test_curvefit_simplecovariance(self):

        def func(x, a, b):
            return a * np.exp(-b*x)

        def jac(x, a, b):
            e = np.exp(-b*x)
            return np.vstack((e, -a * x * e)).T

        np.random.seed(0)
        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3)
        ydata = y + 0.2 * np.random.normal(size=len(xdata))

        sigma = np.zeros(len(xdata)) + 0.2
        covar = np.diag(sigma**2)

        for jac1, jac2 in [(jac, jac), (None, None)]:
            for absolute_sigma in [False, True]:
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma,
                        jac=jac1, absolute_sigma=absolute_sigma)
                popt2, pcov2 = curve_fit(func, xdata, ydata, sigma=covar,
                        jac=jac2, absolute_sigma=absolute_sigma)

                assert_allclose(popt1, popt2, atol=1e-14)
                assert_allclose(pcov1, pcov2, atol=1e-14)
    # 定义一个测试函数，用于测试曲线拟合函数的协方差矩阵计算
    def test_curvefit_covariance(self):

        # 定义第一个拟合函数，包含参数 a 和 b
        def funcp(x, a, b):
            # 定义一个旋转矩阵 rotn
            rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0],
                             [1./np.sqrt(2), 1./np.sqrt(2), 0],
                             [0, 0, 1.0]])
            # 返回 rotn 与 a * exp(-b*x) 的乘积
            return rotn.dot(a * np.exp(-b*x))

        # 定义第一个拟合函数的雅可比矩阵计算函数，包含参数 a 和 b
        def jacp(x, a, b):
            # 定义一个旋转矩阵 rotn
            rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0],
                             [1./np.sqrt(2), 1./np.sqrt(2), 0],
                             [0, 0, 1.0]])
            # 计算 exp(-b*x)
            e = np.exp(-b*x)
            # 返回 rotn 与 [e, -a * x * e]^T 的乘积
            return rotn.dot(np.vstack((e, -a * x * e)).T)

        # 定义第二个拟合函数，只包含参数 a 和 b
        def func(x, a, b):
            # 返回 a * exp(-b*x)
            return a * np.exp(-b*x)

        # 定义第二个拟合函数的雅可比矩阵计算函数，只包含参数 a 和 b
        def jac(x, a, b):
            # 计算 exp(-b*x)
            e = np.exp(-b*x)
            # 返回 [e, -a * x * e]^T
            return np.vstack((e, -a * x * e)).T

        # 设置随机种子
        np.random.seed(0)
        # 生成 x 数据
        xdata = np.arange(1, 4)
        # 计算 y 数据
        y = func(xdata, 2.5, 1.0)
        # 加入正态分布噪声的 y 数据
        ydata = y + 0.2 * np.random.normal(size=len(xdata))
        # 定义每个数据点的误差标准差
        sigma = np.zeros(len(xdata)) + 0.2
        # 计算协方差矩阵
        covar = np.diag(sigma**2)

        # 获取旋转矩阵，然后计算旋转后的 ydata，即 ydatap = R ydata
        rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0],
                         [1./np.sqrt(2), 1./np.sqrt(2), 0],
                         [0, 0, 1.0]])
        ydatap = rotn.dot(ydata)
        # 计算旋转后的协方差矩阵，即 covarp = R C R^T
        covarp = rotn.dot(covar).dot(rotn.T)

        # 循环遍历两个雅可比矩阵计算函数 jac1 和 jac2
        for jac1, jac2 in [(jac, jacp), (None, None)]:
            # 循环遍历是否使用绝对误差标志
            for absolute_sigma in [False, True]:
                # 使用 curve_fit 函数拟合 func 到 xdata 和 ydata，带有指定的 sigma 和雅可比矩阵
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma,
                        jac=jac1, absolute_sigma=absolute_sigma)
                # 使用 curve_fit 函数拟合 funcp 到 xdata 和 ydatap，带有指定的 sigma 和雅可比矩阵
                popt2, pcov2 = curve_fit(funcp, xdata, ydatap, sigma=covarp,
                        jac=jac2, absolute_sigma=absolute_sigma)

                # 断言拟合结果 popt1 和 popt2 非常接近
                assert_allclose(popt1, popt2, rtol=1.2e-7, atol=1e-14)
                # 断言协方差矩阵 pcov1 和 pcov2 非常接近
                assert_allclose(pcov1, pcov2, rtol=1.2e-7, atol=1e-14)

    # 使用参数化测试装饰器，测试带有标量 sigma 的曲线拟合
    @pytest.mark.parametrize("absolute_sigma", [False, True])
    def test_curvefit_scalar_sigma(self, absolute_sigma):
        # 定义线性函数 func(x, a, b) = a * x + b
        def func(x, a, b):
            return a * x + b

        # 获取测试对象的 x 和 y 数据
        x, y = self.x, self.y
        # 使用 curve_fit 函数拟合 func 到 x 和 y，带有标量 sigma 和是否使用绝对误差标志
        _, pcov1 = curve_fit(func, x, y, sigma=2, absolute_sigma=absolute_sigma)
        # 显式构建 sigma 的一维数组
        _, pcov2 = curve_fit(
                func, x, y, sigma=np.full_like(y, 2), absolute_sigma=absolute_sigma
        )
        # 断言两种方法得到的协方差矩阵 pcov1 和 pcov2 相同
        assert np.all(pcov1 == pcov2)
    def test_dtypes(self):
        # 用于回归测试 gh-9581：如果 x 和 y 的数据类型不同，curve_fit 将失败
        x = np.arange(-3, 5)  # 创建一个 numpy 数组 x，包含从 -3 到 4 的整数
        y = 1.5*x + 3.0 + 0.5*np.sin(x)  # 创建一个 numpy 数组 y，按照公式计算其值

        def func(x, a, b):
            return a*x + b  # 定义一个线性函数 func，用于 curve_fit

        for method in ['lm', 'trf', 'dogbox']:  # 遍历优化方法列表
            for dtx in [np.float32, np.float64]:  # 遍历 x 的数据类型列表
                for dty in [np.float32, np.float64]:  # 遍历 y 的数据类型列表
                    x = x.astype(dtx)  # 将 x 转换为指定的数据类型
                    y = y.astype(dty)  # 将 y 转换为指定的数据类型

                with warnings.catch_warnings():  # 捕获警告
                    warnings.simplefilter("error", OptimizeWarning)
                    p, cov = curve_fit(func, x, y, method=method)  # 进行曲线拟合

                    assert np.isfinite(cov).all()  # 检查协方差矩阵中的所有元素是否有限
                    assert not np.allclose(p, 1)   # 检查拟合结果是否与初始值不完全相等

    def test_dtypes2(self):
        # 用于回归测试 gh-7117：如果输入的两个数据都是 float32，curve_fit 将失败
        def hyperbola(x, s_1, s_2, o_x, o_y, c):
            b_2 = (s_1 + s_2) / 2
            b_1 = (s_2 - s_1) / 2
            return o_y + b_1*(x-o_x) + b_2*np.sqrt((x-o_x)**2 + c**2/4)

        min_fit = np.array([-3.0, 0.0, -2.0, -10.0, 0.0])  # 创建一个包含最小适合值的 numpy 数组
        max_fit = np.array([0.0, 3.0, 3.0, 0.0, 10.0])  # 创建一个包含最大适合值的 numpy 数组
        guess = np.array([-2.5/3.0, 4/3.0, 1.0, -4.0, 0.5])  # 创建一个包含初始猜测值的 numpy 数组

        params = [-2, .4, -1, -5, 9.5]  # 创建一个包含超越参数的列表
        xdata = np.array([-32, -16, -8, 4, 4, 8, 16, 32])  # 创建一个 numpy 数组 xdata
        ydata = hyperbola(xdata, *params)  # 根据 hyperbola 函数计算 ydata 的值

        # 运行两次优化，其中 xdata 为 float32 和 float64
        popt_64, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess,
                               bounds=(min_fit, max_fit))

        xdata = xdata.astype(np.float32)  # 将 xdata 转换为 float32 类型
        ydata = hyperbola(xdata, *params)  # 根据 hyperbola 函数重新计算 ydata 的值

        popt_32, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess,
                               bounds=(min_fit, max_fit))

        assert_allclose(popt_32, popt_64, atol=2e-5)  # 检查两次优化的结果是否接近

    def test_broadcast_y(self):
        xdata = np.arange(10)  # 创建一个包含 0 到 9 的整数的 numpy 数组 xdata
        target = 4.7 * xdata ** 2 + 3.5 * xdata + np.random.rand(len(xdata))  # 创建一个目标数组 target
        def fit_func(x, a, b):
            return a * x ** 2 + b * x - target  # 定义一个拟合函数 fit_func

        for method in ['lm', 'trf', 'dogbox']:  # 遍历优化方法列表
            popt0, pcov0 = curve_fit(fit_func,
                                     xdata=xdata,
                                     ydata=np.zeros_like(xdata),  # 将 ydata 初始化为与 xdata 相同大小的零数组
                                     method=method)
            popt1, pcov1 = curve_fit(fit_func,
                                     xdata=xdata,
                                     ydata=0,  # 将 ydata 初始化为标量 0
                                     method=method)
            assert_allclose(pcov0, pcov1)  # 检查两次优化的协方差矩阵是否接近

    def test_args_in_kwargs(self):
        # 确保 `args` 不能作为关键字参数传递给 `curve_fit`

        def func(x, a, b):
            return a * x + b  # 定义一个线性函数 func

        with assert_raises(ValueError):  # 捕获值错误异常
            curve_fit(func,
                      xdata=[1, 2, 3, 4],
                      ydata=[5, 9, 13, 17],
                      p0=[1],
                      args=(1,))  # 尝试使用 `args` 作为关键字参数传递给 curve_fit
    def test_data_point_number_validation(self):
        # 定义一个测试函数，用于验证 curve_fit 是否能正确处理函数参数数量不匹配的情况
        def func(x, a, b, c, d, e):
            return a * np.exp(-b * x) + c + d + e

        # 使用 assert_raises 检查是否会引发 TypeError，并验证错误消息中是否包含特定文本
        with assert_raises(TypeError, match="The number of func parameters="):
            # 调用 curve_fit 函数，传入 xdata 和 ydata，但参数数量不匹配
            curve_fit(func,
                      xdata=[1, 2, 3, 4],
                      ydata=[5, 9, 13, 17])

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_gh4555(self):
        # 测试 gh-4555 报告的问题，即 scipy.optimize.curve_fit 返回的协方差矩阵可能存在负对角元素和特征值问题。
        # 使用自定义的非线性函数 f(x, a, b, c, d, e) 进行拟合
        def f(x, a, b, c, d, e):
            return a*np.log(x + 1 + b) + c*np.log(x + 1 + d) + e

        # 设置随机数生成器，确保结果可重复性
        rng = np.random.default_rng(408113519974467917)
        n = 100
        x = np.arange(n)
        y = np.linspace(2, 7, n) + rng.random(n)
        # 使用 optimize.curve_fit 拟合数据，设置最大迭代次数
        p, cov = optimize.curve_fit(f, x, y, maxfev=100000)
        # 断言协方差矩阵的所有对角元素大于零
        assert np.all(np.diag(cov) > 0)
        # 计算协方差矩阵的特征值，用于进一步调试
        eigs = linalg.eigh(cov)[0]  # separate line for debugging
        # 断言所有特征值均大于一个小负数阈值，用于检查特征值问题是否解决
        assert np.all(eigs > -1e-2)
        # 断言协方差矩阵是对称的
        assert_allclose(cov, cov.T)

    def test_gh4555b(self):
        # 检查 PR gh-17247 对简单情况下协方差矩阵的影响是否显著
        rng = np.random.default_rng(408113519974467917)

        # 定义一个简单的非线性函数 func(x, a, b, c)
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3, 0.5)
        y_noise = 0.2 * rng.normal(size=xdata.size)
        ydata = y + y_noise
        _, res = curve_fit(func, xdata, ydata)
        # 参考值来自提交 1d80a2f254380d2b45733258ca42eb6b55c8755b
        ref = [[+0.0158972536486215, 0.0069207183284242, -0.0007474400714749],
               [+0.0069207183284242, 0.0205057958128679, +0.0053997711275403],
               [-0.0007474400714749, 0.0053997711275403, +0.0027833930320877]]
        # 在默认容差下，使用 assert_allclose 断言结果与参考值接近
        assert_allclose(res, ref, 2e-7)

    def test_gh13670(self):
        # 检查 gh-13670 报告的问题，即 curve_fit 在优化开始时可能会多次调用可调用函数。
        rng = np.random.default_rng(8250058582555444926)
        x = np.linspace(0, 3, 101)
        y = 2 * x + 1 + rng.normal(size=101) * 0.5

        # 定义一个线性函数 line(x, *p)，用于 curve_fit 拟合
        def line(x, *p):
            # 断言当前参数列表与上次调用的不同，以检测优化开始时是否重复调用问题
            assert not np.all(line.last_p == p)
            line.last_p = p
            return x * p[0] + p[1]

        # 定义雅可比矩阵函数 jac(x, *p)
        def jac(x, *p):
            # 断言当前参数列表与上次调用的不同，以检测优化开始时是否重复调用问题
            assert not np.all(jac.last_p == p)
            jac.last_p = p
            return np.array([x, np.ones_like(x)]).T

        line.last_p = None
        jac.last_p = None
        p0 = np.array([1.0, 5.0])
        # 使用 curve_fit 进行拟合，传入起始参数 p0 和自定义的雅可比矩阵 jac
        curve_fit(line, x, y, p0, method='lm', jac=jac)
class TestFixedPoint:

    def test_scalar_trivial(self):
        # 定义一个简单的函数 f(x) = 2x；其固定点应为 x=0
        def func(x):
            return 2.0*x
        x0 = 1.0  # 初始值设定为 1.0
        x = fixed_point(func, x0)  # 调用固定点函数
        assert_almost_equal(x, 0.0)  # 断言计算结果接近 0

    def test_scalar_basic1(self):
        # 定义一个函数 f(x) = x**2；初始值 x0=1.05；其固定点应为 x=1
        def func(x):
            return x**2
        x0 = 1.05  # 初始值设定为 1.05
        x = fixed_point(func, x0)  # 调用固定点函数
        assert_almost_equal(x, 1.0)  # 断言计算结果接近 1.0

    def test_scalar_basic2(self):
        # 定义一个函数 f(x) = x**0.5；初始值 x0=1.05；其固定点应为 x=1
        def func(x):
            return x**0.5
        x0 = 1.05  # 初始值设定为 1.05
        x = fixed_point(func, x0)  # 调用固定点函数
        assert_almost_equal(x, 1.0)  # 断言计算结果接近 1.0

    def test_array_trivial(self):
        # 定义一个函数 f(x) = 2.0*x 作用于数组；初始值 x0=[0.3, 0.15]
        def func(x):
            return 2.0*x
        x0 = [0.3, 0.15]  # 初始值设定为 [0.3, 0.15]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0)  # 调用固定点函数
        assert_almost_equal(x, [0.0, 0.0])  # 断言计算结果接近 [0.0, 0.0]

    def test_array_basic1(self):
        # 定义一个函数 f(x, c) = c * x**2 作用于数组；其固定点应为 x=1/c
        def func(x, c):
            return c * x**2
        c = array([0.75, 1.0, 1.25])  # c 值设定为 [0.75, 1.0, 1.25]
        x0 = [1.1, 1.15, 0.9]  # 初始值设定为 [1.1, 1.15, 0.9]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0, args=(c,))  # 调用固定点函数
        assert_almost_equal(x, 1.0/c)  # 断言计算结果接近 1.0/c

    def test_array_basic2(self):
        # 定义一个函数 f(x, c) = c * x**0.5 作用于数组；其固定点应为 x=c**2
        def func(x, c):
            return c * x**0.5
        c = array([0.75, 1.0, 1.25])  # c 值设定为 [0.75, 1.0, 1.25]
        x0 = [0.8, 1.1, 1.1]  # 初始值设定为 [0.8, 1.1, 1.1]
        x = fixed_point(func, x0, args=(c,))  # 调用固定点函数
        assert_almost_equal(x, c**2)  # 断言计算结果接近 c**2

    def test_lambertw(self):
        # 使用 Lambert W 函数计算的例子
        xxroot = fixed_point(lambda xx: np.exp(-2.0*xx)/2.0, 1.0,
                args=(), xtol=1e-12, maxiter=500)  # 调用固定点函数
        assert_allclose(xxroot, np.exp(-2.0*xxroot)/2.0)  # 断言计算结果接近 np.exp(-2.0*xxroot)/2.0
        assert_allclose(xxroot, lambertw(1)/2)  # 断言计算结果接近 Lambert W 函数的结果除以 2

    def test_no_acceleration(self):
        # 解决 GitHub 上的 issue 5460 的例子
        ks = 2
        kl = 6
        m = 1.3
        n0 = 1.001
        i0 = ((m-1)/m)*(kl/ks/m)**(1/(m-1))

        def func(n):
            return np.log(kl/ks/n) / np.log(i0*n/(n - 1)) + 1

        n = fixed_point(func, n0, method='iteration')  # 使用迭代法调用固定点函数
        assert_allclose(n, m)  # 断言计算结果接近 m
```