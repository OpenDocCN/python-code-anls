# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__basinhopping.py`

```
"""
Unit tests for the basin hopping global minimization algorithm.
"""
# 导入必要的模块和函数
import copy  # 导入copy模块，用于对象的深拷贝操作

# 导入测试相关的函数和类
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)
import pytest  # 导入pytest测试框架
from pytest import raises as assert_raises  # 导入raises别名为assert_raises，用于检查异常

import numpy as np  # 导入NumPy库
from numpy import cos, sin  # 导入cos和sin函数

# 导入SciPy中的优化函数和相关类
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
    Storage, RandomDisplacement, Metropolis, AdaptiveStepsize
)


def func1d(x):
    # 一维函数，返回函数值和导数
    f = cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
    return f, df


def func2d_nograd(x):
    # 二维函数，只返回函数值
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f


def func2d(x):
    # 二维函数，返回函数值和导数
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    df = np.zeros(2)
    df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    df[1] = 2. * x[1] + 0.2
    return f, df


def func2d_easyderiv(x):
    # 简化的二维函数，返回函数值和导数
    f = 2.0*x[0]**2 + 2.0*x[0]*x[1] + 2.0*x[1]**2 - 6.0*x[0]
    df = np.zeros(2)
    df[0] = 4.0*x[0] + 2.0*x[1] - 6.0
    df[1] = 2.0*x[0] + 4.0*x[1]
    return f, df


class MyTakeStep1(RandomDisplacement):
    """自定义步进方法1，继承自RandomDisplacement

    使用一个副本的displace，但设置一个特殊参数以确保其被使用。
    """
    def __init__(self):
        self.been_called = False
        super().__init__()  # 调用父类的构造方法

    def __call__(self, x):
        # 调用实例时执行的方法，设置标志位并调用父类的__call__方法
        self.been_called = True
        return super().__call__(x)


def myTakeStep2(x):
    """自定义步进方法2，与RandomDisplacement相似，但没有stepsize属性

    用函数形式重新实现RandomDisplacement，确保所有功能仍能正常工作。
    """
    s = 0.5
    x += np.random.uniform(-s, s, np.shape(x))
    return x


class MyAcceptTest:
    """自定义接受测试类

    这个类什么也不做，只确保它被使用，并确保所有可能的返回值都被接受。
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0
        self.testres = [False, 'force accept', True, np.bool_(True),
                        np.bool_(False), [], {}, 0, 1]

    def __call__(self, **kwargs):
        # 调用实例时执行的方法，设置标志位并返回测试结果
        self.been_called = True
        self.ncalls += 1
        if self.ncalls - 1 < len(self.testres):
            return self.testres[self.ncalls - 1]
        else:
            return True


class MyCallBack:
    """自定义回调函数类

    确保回调函数被使用，并在10步后返回True以确保提前停止。
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x, f, accepted):
        # 调用实例时执行的方法，设置标志位并在适当时返回True
        self.been_called = True
        self.ncalls += 1
        if self.ncalls == 10:
            return True


class TestBasinHopping:
    """Basin Hopping算法的单元测试类

    这个类用于测试Basin Hopping全局最小化算法的各个功能。
    """
    def setup_method(self):
        """ Tests setup.

        Run tests based on the 1-D and 2-D functions described above.
        """
        # 初始化测试数据
        self.x0 = (1.0, [1.0, 1.0])
        # 预期的解决方案
        self.sol = (-0.195, np.array([-0.195, -0.1]))

        # 允许的误差范围，小数点后3位
        self.tol = 3  # number of decimal places

        # 迭代次数
        self.niter = 100
        # 是否显示详细信息
        self.disp = False

        # 固定随机种子
        np.random.seed(1234)

        # 设置优化方法及参数，包括使用梯度的方法
        self.kwargs = {"method": "L-BFGS-B", "jac": True}
        # 设置优化方法及参数，不使用梯度的方法
        self.kwargs_nograd = {"method": "L-BFGS-B"}

    def test_TypeError(self):
        # 测试当输入错误时是否引发 TypeError 异常
        i = 1
        # 如果 take_step 参数不是可调用对象，应该引发 TypeError
        assert_raises(TypeError, basinhopping, func2d, self.x0[i],
                      take_step=1)
        # 如果 accept_test 参数不是可调用对象，应该引发 TypeError
        assert_raises(TypeError, basinhopping, func2d, self.x0[i],
                      accept_test=1)

    def test_input_validation(self):
        # 测试输入参数的有效性验证
        msg = 'target_accept_rate has to be in range \\(0, 1\\)'
        # 当 target_accept_rate 不在 (0, 1) 范围内时，应引发 ValueError 异常
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=0.)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], target_accept_rate=1.)

        msg = 'stepwise_factor has to be in range \\(0, 1\\)'
        # 当 stepwise_factor 不在 (0, 1) 范围内时，应引发 ValueError 异常
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=0.)
        with assert_raises(ValueError, match=msg):
            basinhopping(func1d, self.x0[0], stepwise_factor=1.)

    def test_1d_grad(self):
        # 测试带有梯度的一维最小化
        i = 0
        # 执行基于 self.kwargs 参数的基本局部优化方法的全局优化
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        # 检查计算结果是否接近预期解决方案，给定的误差范围
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d(self):
        # 测试带有梯度的二维最小化
        i = 1
        # 执行基于 self.kwargs 参数的基本局部优化方法的全局优化
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        # 检查计算结果是否接近预期解决方案，给定的误差范围
        assert_almost_equal(res.x, self.sol[i], self.tol)
        # 确保计算的函数评估次数大于零
        assert_(res.nfev > 0)

    def test_njev(self):
        # 测试是否正确返回 njev
        i = 1
        # 复制 self.kwargs，并修改方法为 "BFGS"，因为 L-BFGS-B 不使用 njev
        minimizer_kwargs = self.kwargs.copy()
        minimizer_kwargs["method"] = "BFGS"
        # 执行基于修改后的 minimizer_kwargs 参数的全局优化
        res = basinhopping(func2d, self.x0[i],
                           minimizer_kwargs=minimizer_kwargs, niter=self.niter,
                           disp=self.disp)
        # 确保计算的函数评估次数大于零
        assert_(res.nfev > 0)
        # 确保 nfev 与 njev 相等
        assert_equal(res.nfev, res.njev)
    def test_jac(self):
        # test Jacobian returned
        # 复制参数字典以确保不改变原始参数
        minimizer_kwargs = self.kwargs.copy()
        # 使用BFGS方法进行最小化，该方法返回雅可比矩阵
        minimizer_kwargs["method"] = "BFGS"

        # 在多次迭代中应用基于跳跃的全局最优化算法
        res = basinhopping(func2d_easyderiv, [0.0, 0.0],
                           minimizer_kwargs=minimizer_kwargs, niter=self.niter,
                           disp=self.disp)

        # 断言最优化结果对象具有“jac”属性
        assert_(hasattr(res.lowest_optimization_result, "jac"))

        # 在这种情况下，雅可比矩阵为 [df/dx, df/dy]
        _, jacobian = func2d_easyderiv(res.x)
        # 断言最优化结果中的雅可比矩阵与预期的一致
        assert_almost_equal(res.lowest_optimization_result.jac, jacobian,
                            self.tol)

    def test_2d_nograd(self):
        # test 2-D minimizations without gradient
        # 选择测试用例中的索引
        i = 1
        # 在没有梯度的情况下应用基于跳跃的全局最优化算法
        res = basinhopping(func2d_nograd, self.x0[i],
                           minimizer_kwargs=self.kwargs_nograd,
                           niter=self.niter, disp=self.disp)
        # 断言结果与预期解几乎相等
        assert_almost_equal(res.x, self.sol[i], self.tol)

    @pytest.mark.fail_slow(10)
    def test_all_minimizers(self):
        # Test 2-D minimizations with gradient. Nelder-Mead, Powell, COBYLA, and
        # COBYQA don't accept jac=True, so aren't included here.
        # 选择测试用例中的索引
        i = 1
        # 定义所有可用最小化方法
        methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP']
        # 复制参数字典以确保不改变原始参数
        minimizer_kwargs = copy.copy(self.kwargs)
        for method in methods:
            minimizer_kwargs["method"] = method
            # 应用基于跳跃的全局最优化算法，使用指定的最小化方法
            res = basinhopping(func2d, self.x0[i],
                               minimizer_kwargs=minimizer_kwargs,
                               niter=self.niter, disp=self.disp)
            # 断言结果与预期解几乎相等
            assert_almost_equal(res.x, self.sol[i], self.tol)

    @pytest.mark.fail_slow(20)
    def test_all_nograd_minimizers(self):
        # Test 2-D minimizations without gradient. Newton-CG requires jac=True,
        # so not included here.
        # 选择测试用例中的索引
        i = 1
        # 定义所有可用无梯度最小化方法
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP',
                   'Nelder-Mead', 'Powell', 'COBYLA', 'COBYQA']
        # 复制参数字典以确保不改变原始参数
        minimizer_kwargs = copy.copy(self.kwargs_nograd)
        for method in methods:
            # COBYQA方法在此问题上需要大量时间
            niter = 10 if method == 'COBYQA' else self.niter
            minimizer_kwargs["method"] = method
            # 应用基于跳跃的全局最优化算法，使用指定的最小化方法
            res = basinhopping(func2d_nograd, self.x0[i],
                               minimizer_kwargs=minimizer_kwargs,
                               niter=niter, disp=self.disp)
            tol = self.tol
            # 对于COBYLA方法，使用更宽松的精度
            if method == 'COBYLA':
                tol = 2
            # 断言结果与预期解几乎相等，使用指定的精度
            assert_almost_equal(res.x, self.sol[i], decimal=tol)
    # 定义一个测试函数，用于测试自定义的takestep是否正常工作
    def test_pass_takestep(self):
        # 创建一个MyTakeStep1的实例作为自定义的takestep对象
        takestep = MyTakeStep1()
        # 记录初始步长大小
        initial_step_size = takestep.stepsize
        # 初始化索引为1
        i = 1
        # 运行basinhopping优化，传入函数func2d、起始点self.x0[i]、minimizer_kwargs参数和其他设置
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp,
                           take_step=takestep)
        # 断言优化结果res的最优点res.x与预期解self.sol[i]之间的近似相等关系，精度为self.tol
        assert_almost_equal(res.x, self.sol[i], self.tol)
        # 断言takestep是否被调用过
        assert_(takestep.been_called)
        # 断言初始步长大小与优化后的步长大小不同，确保内置的自适应步长大小机制被使用
        assert_(initial_step_size != takestep.stepsize)

    # 定义一个测试函数，用于测试不带stepsize属性的自定义takestep是否正常工作
    def test_pass_simple_takestep(self):
        # 创建一个myTakeStep2的实例作为自定义的takestep对象
        takestep = myTakeStep2
        # 初始化索引为1
        i = 1
        # 运行basinhopping优化，传入函数func2d_nograd、起始点self.x0[i]、minimizer_kwargs参数和其他设置
        res = basinhopping(func2d_nograd, self.x0[i],
                           minimizer_kwargs=self.kwargs_nograd,
                           niter=self.niter, disp=self.disp,
                           take_step=takestep)
        # 断言优化结果res的最优点res.x与预期解self.sol[i]之间的近似相等关系，精度为self.tol
        assert_almost_equal(res.x, self.sol[i], self.tol)

    # 定义一个测试函数，用于测试自定义的accept test是否正常工作
    def test_pass_accept_test(self):
        # 创建一个MyAcceptTest的实例作为自定义的accept test对象
        accept_test = MyAcceptTest()
        # 初始化索引为1
        i = 1
        # 运行basinhopping优化，传入函数func2d、起始点self.x0[i]、minimizer_kwargs参数和其他设置
        # 限制最大迭代次数为10次
        basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                     niter=10, disp=self.disp, accept_test=accept_test)
        # 断言accept test是否被调用过
        assert_(accept_test.been_called)

    # 定义一个测试函数，用于测试自定义的callback函数是否正常工作
    def test_pass_callback(self):
        # 创建一个MyCallBack的实例作为自定义的callback对象
        callback = MyCallBack()
        # 初始化索引为1
        i = 1
        # 运行basinhopping优化，传入函数func2d、起始点self.x0[i]、minimizer_kwargs参数和其他设置
        # 限制最大迭代次数为30次
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=30, disp=self.disp, callback=callback)
        # 断言callback是否被调用过
        assert_(callback.been_called)
        # 断言优化结果res的消息中包含"callback"，表明callback函数被正确使用
        assert_("callback" in res.message[0])
        # 断言MyCallBack在BasinHoppingRunner构造期间被调用次数为1，因此在MyCallBack停止优化前，还剩余9次迭代
        assert_equal(res.nit, 9)

    # 定义一个测试函数，用于测试当优化器失败时的情况
    def test_minimizer_fail(self):
        # 初始化索引为1
        i = 1
        # 设定最大迭代次数为0
        self.kwargs["options"] = dict(maxiter=0)
        self.niter = 10
        # 运行basinhopping优化，传入函数func2d、起始点self.x0[i]、minimizer_kwargs参数和其他设置
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        # 断言失败的最小化次数应该等于迭代次数加1
        assert_equal(res.nit + 1, res.minimization_failures)
    def test_niter_zero(self):
        # 测试当 niter=0 时的行为，来自 gh5915
        i = 0
        # 使用 basinhopping 函数进行优化，但迭代次数 niter=0
        basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
                     niter=0, disp=self.disp)

    def test_seed_reproducibility(self):
        # seed 应确保在不同运行中的可重现性
        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}

        f_1 = []

        def callback(x, f, accepted):
            f_1.append(f)

        # 使用 basinhopping 函数进行优化，设置 niter=10，callback 函数和 seed=10
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs,
                     niter=10, callback=callback, seed=10)

        f_2 = []

        def callback2(x, f, accepted):
            f_2.append(f)

        # 再次使用 basinhopping 函数进行优化，确保与前一次相同的 seed=10
        basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs,
                     niter=10, callback=callback2, seed=10)
        # 断言两次优化过程中记录的函数值数组 f_1 和 f_2 相等
        assert_equal(np.array(f_1), np.array(f_2))

    def test_random_gen(self):
        # 检查 np.random.Generator 是否可用（numpy 版本 >= 1.17）
        rng = np.random.default_rng(1)

        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}

        # 使用 basinhopping 函数进行优化，使用指定的随机数生成器 rng，并设置 niter=10
        res1 = basinhopping(func2d, [1.0, 1.0],
                            minimizer_kwargs=minimizer_kwargs,
                            niter=10, seed=rng)

        rng = np.random.default_rng(1)
        # 再次使用相同的随机数生成器 rng 进行优化，确保结果可重现
        res2 = basinhopping(func2d, [1.0, 1.0],
                            minimizer_kwargs=minimizer_kwargs,
                            niter=10, seed=rng)
        # 断言两次优化的结果 res1.x 和 res2.x 相等
        assert_equal(res1.x, res2.x)

    def test_monotonic_basin_hopping(self):
        # 测试带有梯度和温度 T=0 的一维最小化
        i = 0
        # 使用 basinhopping 函数进行优化，设置迭代次数为 self.niter，温度 T=0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp, T=0)
        # 断言优化得到的结果 res.x 接近于预期的 self.sol[i]，误差为 self.tol
        assert_almost_equal(res.x, self.sol[i], self.tol)
class Test_Storage:
    # 设置每个测试方法的初始化条件
    def setup_method(self):
        # 初始化一个包含单个元素的 NumPy 数组
        self.x0 = np.array(1)
        # 初始化初始函数值为 0
        self.f0 = 0

        # 创建一个成功的优化结果对象
        minres = OptimizeResult(success=True)
        # 设置优化结果对象的最优参数为 self.x0
        minres.x = self.x0
        # 设置优化结果对象的最优函数值为 self.f0
        minres.fun = self.f0

        # 初始化 Storage 类实例，传入最优结果对象 minres
        self.storage = Storage(minres)

    # 测试更高函数值是否被拒绝
    def test_higher_f_rejected(self):
        # 创建一个成功的新优化结果对象
        new_minres = OptimizeResult(success=True)
        # 设置新优化结果对象的最优参数为 self.x0 + 1
        new_minres.x = self.x0 + 1
        # 设置新优化结果对象的最优函数值为 self.f0 + 1
        new_minres.fun = self.f0 + 1

        # 更新存储器中的数据，并记录返回值
        ret = self.storage.update(new_minres)
        # 获取存储器中的最低函数值优化结果对象
        minres = self.storage.get_lowest()
        # 断言存储器中的最低参数与 self.x0 相等
        assert_equal(self.x0, minres.x)
        # 断言存储器中的最低函数值与 self.f0 相等
        assert_equal(self.f0, minres.fun)
        # 断言返回值为假（新函数值较高，被拒绝）
        assert_(not ret)

    # 参数化测试，测试更低函数值是否被接受
    @pytest.mark.parametrize('success', [True, False])
    def test_lower_f_accepted(self, success):
        # 创建一个具有指定成功状态的新优化结果对象
        new_minres = OptimizeResult(success=success)
        # 设置新优化结果对象的最优参数为 self.x0 + 1
        new_minres.x = self.x0 + 1
        # 设置新优化结果对象的最优函数值为 self.f0 - 1
        new_minres.fun = self.f0 - 1

        # 更新存储器中的数据，并记录返回值
        ret = self.storage.update(new_minres)
        # 获取存储器中的最低函数值优化结果对象
        minres = self.storage.get_lowest()
        # 断言存储器中的最低参数是否与 self.x0 不相等的布尔结果与 success 一致
        assert (self.x0 != minres.x) == success
        # 断言存储器中的最低函数值是否与 self.f0 不相等的布尔结果与 success 一致
        assert (self.f0 != minres.fun) == success
        # 断言返回值与 success 一致
        assert ret is success


class Test_RandomDisplacement:
    # 设置每个测试方法的初始化条件
    def setup_method(self):
        # 设置步长为 1.0
        self.stepsize = 1.0
        # 创建 RandomDisplacement 类实例，传入步长参数
        self.displace = RandomDisplacement(stepsize=self.stepsize)
        # 设置数组长度为 300000，初始化全零 NumPy 数组
        self.N = 300000
        self.x0 = np.zeros([self.N])

    # 测试随机位移函数
    def test_random(self):
        # 测试均值应为 0
        # 测试方差应为 (2*stepsize)**2 / 12
        # 注意这些测试是随机的，有时会失败
        x = self.displace(self.x0)
        v = (2. * self.stepsize) ** 2 / 12
        # 断言数组 x 的均值接近 0，精度为小数点后一位
        assert_almost_equal(np.mean(x), 0., 1)
        # 断言数组 x 的方差接近 v，精度为小数点后一位
        assert_almost_equal(np.var(x), v, 1)


class Test_Metropolis:
    # 设置每个测试方法的初始化条件
    def setup_method(self):
        # 设置温度为 2.0
        self.T = 2.
        # 创建 Metropolis 类实例，传入温度参数
        self.met = Metropolis(self.T)
        # 创建一个成功的新优化结果对象，函数值为 0
        self.res_new = OptimizeResult(success=True, fun=0.)
        # 创建一个成功的旧优化结果对象，函数值为 1
        self.res_old = OptimizeResult(success=True, fun=1.)

    # 测试返回值必须是布尔型，否则将在 basinhopping 中引发错误
    def test_boolean_return(self):
        # 执行 Metropolis 类中的接受/拒绝测试，并记录返回值
        ret = self.met(res_new=self.res_new, res_old=self.res_old)
        # 断言返回值为布尔型
        assert isinstance(ret, bool)

    # 测试接受更低函数值的情况
    def test_lower_f_accepted(self):
        # 断言 Metropolis 类中的接受/拒绝测试为真
        assert_(self.met(res_new=self.res_new, res_old=self.res_old))

    # 测试对于 f_new > f_old 时，步骤是否随机接受
    def test_accept(self):
        # 测试步骤随机接受 f_new > f_old 的情况
        one_accept = False
        one_reject = False
        for i in range(1000):
            if one_accept and one_reject:
                break
            # 创建一个成功的新优化结果对象，函数值为 1
            res_new = OptimizeResult(success=True, fun=1.)
            # 创建一个成功的旧优化结果对象，函数值为 0.5
            res_old = OptimizeResult(success=True, fun=0.5)
            # 执行 Metropolis 类中的接受/拒绝测试，并记录返回值
            ret = self.met(res_new=res_new, res_old=res_old)
            if ret:
                one_accept = True
            else:
                one_reject = True
        # 断言至少有一个步骤被接受
        assert_(one_accept)
        # 断言至少有一个步骤被拒绝
        assert_(one_reject)
    # 定义一个测试方法，用于测试 GH7495 的问题
    def test_GH7495(self):
        # 在 exp 函数中可能出现溢出，导致 RuntimeWarning
        # 创建一个 Metropolis 对象，以确保在测试过程中不受 self.T 更改的影响
        met = Metropolis(2)
        # 创建两个 OptimizeResult 对象，一个是成功的情况下的结果，另一个是失败的情况下的结果
        res_new = OptimizeResult(success=True, fun=0.)
        res_old = OptimizeResult(success=True, fun=2000)
        # 设置 numpy 的错误状态，当发生溢出时抛出异常
        with np.errstate(over='raise'):
            # 调用 Metropolis 对象的 accept_reject 方法，传入两个结果对象作为参数
            met.accept_reject(res_new=res_new, res_old=res_old)

    # 定义一个测试方法，用于验证修复了 GH7799 报告的问题
    def test_gh7799(self):
        # 定义一个函数 func，用于计算特定公式的值
        def func(x):
            return (x**2-8)**2+(x+2)**2

        x0 = -4
        limit = 50  # 设置函数值的下限为 50
        # 定义一个约束条件，确保函数值不小于 limit
        con = {'type': 'ineq', 'fun': lambda x: func(x) - limit},
        # 使用 basinhopping 函数进行全局优化，验证结果是否成功并且函数值接近于 limit
        res = basinhopping(func, x0, 30, minimizer_kwargs={'constraints': con})
        # 断言优化是否成功
        assert res.success
        # 断言优化后的函数值是否接近于 limit，给定一个相对容差
        assert_allclose(res.fun, limit, rtol=1e-6)

    # 定义一个测试方法，验证 Metropolis 对象在不同情况下的接受行为
    def test_accept_gh7799(self):
        # 创建一个 Metropolis 对象，用于测试
        met = Metropolis(0)  # monotonic basin hopping
        # 创建两个 OptimizeResult 对象，一个是成功的情况下的结果，另一个是失败的情况下的结果
        res_new = OptimizeResult(success=True, fun=0.)
        res_old = OptimizeResult(success=True, fun=1.)

        # 如果新的局部搜索成功且能量更低，则应接受新结果
        assert met(res_new=res_new, res_old=res_old)
        # 如果新的局部搜索不成功，则不应接受新结果，即使能量更低
        res_new.success = False
        assert not met(res_new=res_new, res_old=res_old)
        # ...除非旧的搜索结果也不成功。在这种情况下，可以接受新结果
        res_old.success = False
        assert met(res_new=res_new, res_old=res_old)

    # 定义一个测试方法，验证在没有可行解的情况下的行为
    def test_reject_all_gh7799(self):
        # 定义一个简单的函数 fun，计算向量 x 的平方和
        def fun(x):
            return x @ x

        # 定义一个约束条件，简单地返回 x + 1
        def constraint(x):
            return x + 1

        # 设置优化参数
        kwargs = {'constraints': {'type': 'eq', 'fun': constraint},
                  'bounds': [(0, 1), (0, 1)], 'method': 'slsqp'}
        # 使用 basinhopping 函数进行优化，验证结果应该是不成功的
        res = basinhopping(fun, x0=[2, 3], niter=10, minimizer_kwargs=kwargs)
        # 断言结果不应该成功
        assert not res.success
class Test_AdaptiveStepsize:
    # 设置每个测试方法运行前的初始化方法
    def setup_method(self):
        # 初始化步长为1.0
        self.stepsize = 1.
        # 创建 RandomDisplacement 实例，使用设定的步长
        self.ts = RandomDisplacement(stepsize=self.stepsize)
        # 目标接受率设定为0.5
        self.target_accept_rate = 0.5
        # 创建 AdaptiveStepsize 实例，使用 RandomDisplacement 实例作为 takestep 参数，
        # 关闭详细信息输出，设定目标接受率
        self.takestep = AdaptiveStepsize(takestep=self.ts, verbose=False,
                                         accept_rate=self.target_accept_rate)

    # 测试当接受率低时步长是否增加
    def test_adaptive_increase(self):
        # 初始值设定为0
        x = 0.
        # 运行 AdaptiveStepsize 实例的 takestep 方法
        self.takestep(x)
        # 输出不接受报告
        self.takestep.report(False)
        # 循环执行 takestep 方法，报告接受结果
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(True)
        # 断言当前步长大于初始步长
        assert_(self.ts.stepsize > self.stepsize)

    # 测试当接受率高时步长是否减少
    def test_adaptive_decrease(self):
        # 初始值设定为0
        x = 0.
        # 运行 AdaptiveStepsize 实例的 takestep 方法
        self.takestep(x)
        # 输出接受报告
        self.takestep.report(True)
        # 循环执行 takestep 方法，报告接受结果
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(False)
        # 断言当前步长小于初始步长
        assert_(self.ts.stepsize < self.stepsize)

    # 测试所有步骤均被接受时的情况
    def test_all_accepted(self):
        # 初始值设定为0
        x = 0.
        # 循环执行 takestep 方法，报告接受结果
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(True)
        # 断言当前步长大于初始步长
        assert_(self.ts.stepsize > self.stepsize)

    # 测试所有步骤均被拒绝时的情况
    def test_all_rejected(self):
        # 初始值设定为0
        x = 0.
        # 循环执行 takestep 方法，报告接受结果
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(False)
        # 断言当前步长小于初始步长
        assert_(self.ts.stepsize < self.stepsize)
```