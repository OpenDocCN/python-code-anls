# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_direct.py`

```
"""
Unit test for DIRECT optimization algorithm.
"""
# 导入必要的库和模块
from numpy.testing import (assert_allclose,
                           assert_array_less)
import pytest
import numpy as np
from scipy.optimize import direct, Bounds

# 测试类
class TestDIRECT:

    # 设置方法，初始化测试数据
    def setup_method(self):
        self.fun_calls = 0  # 记录函数调用次数
        self.bounds_sphere = 4*[(-2, 3)]  # 球形函数的边界
        self.optimum_sphere_pos = np.zeros((4, ))  # 球形函数的最优位置
        self.optimum_sphere = 0.0  # 球形函数的最优值
        self.bounds_stylinski_tang = Bounds([-4., -4.], [4., 4.])  # Stylinski-Tang函数的边界
        self.maxiter = 1000  # 最大迭代次数

    # 球形函数
    def sphere(self, x):
        self.fun_calls += 1  # 记录函数调用次数
        return np.square(x).sum()

    # 除法函数
    def inv(self, x):
        if np.sum(x) == 0:
            raise ZeroDivisionError()
        return 1/np.sum(x)

    # 返回NaN的函数
    def nan_fun(self, x):
        return np.nan

    # 返回无穷大的函数
    def inf_fun(self, x):
        return np.inf

    # Stylinski-Tang函数
    def styblinski_tang(self, pos):
        x, y = pos
        return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)

    # 参数化测试：DIRECT优化算法
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_direct(self, locally_biased):
        # 运行DIRECT优化算法
        res = direct(self.sphere, self.bounds_sphere,
                     locally_biased=locally_biased)

        # 测试结果的准确性
        assert_allclose(res.x, self.optimum_sphere_pos,
                        rtol=1e-3, atol=1e-3)
        assert_allclose(res.fun, self.optimum_sphere, atol=1e-5, rtol=1e-5)

        # 测试结果是否在边界内
        _bounds = np.asarray(self.bounds_sphere)
        assert_array_less(_bounds[:, 0], res.x)
        assert_array_less(res.x, _bounds[:, 1])

        # 测试函数评估次数是否符合预期。原始DIRECT算法最后一次迭代可能超过500次评估
        assert res.nfev <= 1000 * (len(self.bounds_sphere) + 1)
        # 测试函数评估次数是否正确
        assert res.nfev == self.fun_calls

        # 测试迭代次数是否在最大迭代次数之内
        assert res.nit <= self.maxiter

    # 参数化测试：DIRECT优化算法
    @pytest.mark.parametrize("locally_biased", [True, False])
    # 定义测试方法，用于测试直接调用时回调函数不改变结果的情况
    def test_direct_callback(self, locally_biased):
        # 调用 direct 函数，不使用回调函数，获取结果
        res = direct(self.sphere, self.bounds_sphere,
                     locally_biased=locally_biased)

        # 定义回调函数 callback，对输入值进行操作并返回平方值
        def callback(x):
            x = 2*x  # 输入值乘以2
            dummy = np.square(x)  # 计算输入值的平方
            print("DIRECT minimization algorithm callback test")  # 打印测试信息
            return dummy  # 返回计算结果

        # 调用 direct 函数，使用定义的回调函数 callback，获取结果
        res_callback = direct(self.sphere, self.bounds_sphere,
                              locally_biased=locally_biased,
                              callback=callback)

        # 断言两次 direct 调用的结果的最优解 x 相等
        assert_allclose(res.x, res_callback.x)

        # 断言两次 direct 调用的迭代次数相等
        assert res.nit == res_callback.nit
        # 断言两次 direct 调用的函数评估次数相等
        assert res.nfev == res_callback.nfev
        # 断言两次 direct 调用的运行状态相等
        assert res.status == res_callback.status
        # 断言两次 direct 调用的成功状态相等
        assert res.success == res_callback.success
        # 断言两次 direct 调用的最优函数值相等
        assert res.fun == res_callback.fun
        # 再次断言两次 direct 调用的最优解 x 相等
        assert_allclose(res.x, res_callback.x)
        # 断言两次 direct 调用的消息字符串相等
        assert res.message == res_callback.message

        # 测试结果的准确性，断言回调函数结果的最优解 x 接近预期的最优位置
        assert_allclose(res_callback.x, self.optimum_sphere_pos,
                        rtol=1e-3, atol=1e-3)
        # 断言回调函数结果的最优函数值接近预期的最优值
        assert_allclose(res_callback.fun, self.optimum_sphere,
                        atol=1e-5, rtol=1e-5)

    # 使用 pytest 的参数化装饰器，测试在特定条件下会引发 ZeroDivisionError 异常的情况
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_exception(self, locally_biased):
        # 定义边界条件
        bounds = 4*[(-10, 10)]
        # 使用 pytest 断言检查是否引发 ZeroDivisionError 异常
        with pytest.raises(ZeroDivisionError):
            direct(self.inv, bounds=bounds,
                   locally_biased=locally_biased)

    # 使用 pytest 的参数化装饰器，测试函数中存在 NaN 值的情况
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_nan(self, locally_biased):
        # 定义边界条件
        bounds = 4*[(-10, 10)]
        # 调用 direct 函数，测试函数中存在 NaN 值的情况
        direct(self.nan_fun, bounds=bounds,
               locally_biased=locally_biased)

    # 使用 pytest 的多重参数化装饰器，测试在特定的长度容差条件下的情况
    @pytest.mark.parametrize("len_tol", [1e-3, 1e-4])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_len_tol(self, len_tol, locally_biased):
        # 定义边界条件
        bounds = 4*[(-10., 10.)]
        # 调用 direct 函数，测试在给定长度容差下的情况
        res = direct(self.sphere, bounds=bounds, len_tol=len_tol,
                     vol_tol=1e-30, locally_biased=locally_biased)
        # 断言结果状态为 5
        assert res.status == 5
        # 断言结果为成功状态
        assert res.success
        # 断言最优解 x 接近全零向量
        assert_allclose(res.x, np.zeros((4, )))
        # 消息字符串包含长度容差信息
        message = ("The side length measure of the hyperrectangle containing "
                   "the lowest function value found is below "
                   f"len_tol={len_tol}")
        # 断言结果消息与预期消息相符
        assert res.message == message

    # 使用 pytest 的多重参数化装饰器，测试在特定的体积容差条件下的情况
    @pytest.mark.parametrize("vol_tol", [1e-6, 1e-8])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_vol_tol(self, vol_tol, locally_biased):
        # 定义边界条件
        bounds = 4*[(-10., 10.)]
        # 调用 direct 函数，测试在给定体积容差下的情况
        res = direct(self.sphere, bounds=bounds, vol_tol=vol_tol,
                     len_tol=0., locally_biased=locally_biased)
        # 断言结果状态为 4
        assert res.status == 4
        # 断言结果为成功状态
        assert res.success
        # 断言最优解 x 接近全零向量
        assert_allclose(res.x, np.zeros((4, )))
        # 消息字符串包含体积容差信息
        message = ("The volume of the hyperrectangle containing the lowest "
                   f"function value found is below vol_tol={vol_tol}")
        # 断言结果消息与预期消息相符
        assert res.message == message
    # 使用 pytest 的 parametrize 装饰器，指定多组参数进行测试
    @pytest.mark.parametrize("f_min_rtol", [1e-3, 1e-5, 1e-7])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_f_min(self, f_min_rtol, locally_biased):
        # 测试确保在 f_min_rtol 的相对误差范围内达到期望的函数值
        f_min = 1.
        bounds = 4*[(-2., 10.)]
        # 调用 direct 函数进行优化，设置参数和边界条件
        res = direct(self.sphere, bounds=bounds, f_min=f_min,
                     f_min_rtol=f_min_rtol,
                     locally_biased=locally_biased)
        # 断言优化结果的状态为 3（已收敛）
        assert res.status == 3
        # 断言优化成功
        assert res.success
        # 断言找到的最优函数值小于期望最小函数值的 (1 + f_min_rtol) 倍
        assert res.fun < f_min * (1. + f_min_rtol)
        # 构建消息字符串，说明找到的最优函数值与全局最优值 f_min 的相对误差
        message = ("The best function value found is within a relative "
                   f"error={f_min_rtol} of the (known) global optimum f_min")
        # 断言优化结果的消息与预期消息相符
        assert res.message == message

    def circle_with_args(self, x, a, b):
        # 定义一个简单的二维圆函数，用于优化测试
        return np.square(x[0] - a) + np.square(x[1] - b).sum()

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_f_circle_with_args(self, locally_biased):
        bounds = 2*[(-2.0, 2.0)]
        # 调用 direct 函数进行优化，设置参数和边界条件，传递额外参数 args=(1, 1)，最大函数评估次数为 1250
        res = direct(self.circle_with_args, bounds, args=(1, 1), maxfun=1250,
                     locally_biased=locally_biased)
        # 断言优化结果的最优解与预期解 [1., 1.] 的相对误差不超过 1e-5
        assert_allclose(res.x, np.array([1., 1.]), rtol=1e-5)

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_failure_maxfun(self, locally_biased):
        # 测试当优化运行达到最大函数评估次数时，是否返回 success = False
        maxfun = 100
        # 调用 direct 函数进行优化，设置参数和边界条件，设置最大函数评估次数 maxfun
        result = direct(self.styblinski_tang, self.bounds_stylinski_tang,
                        maxfun=maxfun, locally_biased=locally_biased)
        # 断言优化结果的成功标志为 False
        assert result.success is False
        # 断言优化结果的状态为 1（达到最大函数评估次数）
        assert result.status == 1
        # 断言实际的函数评估次数大于等于设定的最大评估次数 maxfun
        assert result.nfev >= maxfun
        # 构建消息字符串，说明函数评估次数超过设定的最大值 maxfun
        message = ("Number of function evaluations done is "
                   f"larger than maxfun={maxfun}")
        # 断言优化结果的消息与预期消息相符
        assert result.message == message

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_failure_maxiter(self, locally_biased):
        # 测试当优化运行达到最大迭代次数时，是否返回 success = False
        maxiter = 10
        # 调用 direct 函数进行优化，设置参数和边界条件，设置最大迭代次数 maxiter
        result = direct(self.styblinski_tang, self.bounds_stylinski_tang,
                        maxiter=maxiter, locally_biased=locally_biased)
        # 断言优化结果的成功标志为 False
        assert result.success is False
        # 断言优化结果的状态为 2（达到最大迭代次数）
        assert result.status == 2
        # 断言实际的迭代次数大于等于设定的最大迭代次数 maxiter
        assert result.nit >= maxiter
        # 构建消息字符串，说明迭代次数超过设定的最大值 maxiter
        message = f"Number of iterations is larger than maxiter={maxiter}"
        # 断言优化结果的消息与预期消息相符
        assert result.message == message

    @pytest.mark.parametrize("locally_biased", [True, False])
    # 测试不同边界表示方式对结果的影响

    # 定义旧边界的下限和上限
    lb = [-6., 1., -5.]
    ub = [-1., 3., 5.]

    # 定义优化变量的理论最优解
    x_opt = np.array([-1., 1., 0.])

    # 将旧边界打包成元组列表
    bounds_old = list(zip(lb, ub))

    # 使用Bounds类创建新的边界对象
    bounds_new = Bounds(lb, ub)

    # 使用DIRECT算法分别在旧边界和新边界上进行优化
    res_old_bounds = direct(self.sphere, bounds_old,
                            locally_biased=locally_biased)
    res_new_bounds = direct(self.sphere, bounds_new,
                            locally_biased=locally_biased)

    # 检查新旧边界下优化的结果是否一致
    assert res_new_bounds.nfev == res_old_bounds.nfev
    assert res_new_bounds.message == res_old_bounds.message
    assert res_new_bounds.success == res_old_bounds.success
    assert res_new_bounds.nit == res_old_bounds.nit
    assert_allclose(res_new_bounds.x, res_old_bounds.x)
    
    # 检查新边界下的优化结果是否接近预期的最优解
    assert_allclose(res_new_bounds.x, x_opt, rtol=1e-2)
    # 测试函数，用于验证 f_min_rtol 参数的有效性
    def test_fmin_rtol_validation(self, f_min_rtol):
        # 定义错误消息
        error_msg = "f_min_rtol must be between 0 and 1."
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 f_min_rtol，并指定 f_min=0.
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   f_min_rtol=f_min_rtol, f_min=0.)

    # 使用 pytest 参数化装饰器标记，验证 maxfun 参数的类型错误
    @pytest.mark.parametrize("maxfun", [1.5, "string", (1, 2)])
    def test_maxfun_wrong_type(self, maxfun):
        # 定义错误消息
        error_msg = "maxfun must be of type int."
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 maxfun
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxfun=maxfun)

    # 使用 pytest 参数化装饰器标记，验证 maxiter 参数的类型错误
    @pytest.mark.parametrize("maxiter", [1.5, "string", (1, 2)])
    def test_maxiter_wrong_type(self, maxiter):
        # 定义错误消息
        error_msg = "maxiter must be of type int."
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 maxiter
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxiter=maxiter)

    # 测试函数，验证 maxiter 参数为负数时的错误
    def test_negative_maxiter(self):
        # 定义错误消息
        error_msg = "maxiter must be > 0."
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 maxiter=-1
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxiter=-1)

    # 测试函数，验证 maxfun 参数为负数时的错误
    def test_negative_maxfun(self):
        # 定义错误消息
        error_msg = "maxfun must be > 0."
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 maxfun=-1
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxfun=-1)

    # 使用 pytest 参数化装饰器标记，验证 bounds 参数的类型错误
    @pytest.mark.parametrize("bounds", ["bounds", 2., 0])
    def test_invalid_bounds_type(self, bounds):
        # 定义错误消息
        error_msg = ("bounds must be a sequence or "
                     "instance of Bounds class")
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 bounds
            direct(self.styblinski_tang, bounds)

    # 使用 pytest 参数化装饰器标记，验证 bounds 参数的不一致性错误
    @pytest.mark.parametrize("bounds",
                             [Bounds([-1., -1], [-2, 1]),
                              Bounds([-np.nan, -1], [-2, np.nan]),
                              ]
                             )
    def test_incorrect_bounds(self, bounds):
        # 定义错误消息
        error_msg = 'Bounds are not consistent min < max'
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 bounds
            direct(self.styblinski_tang, bounds)

    # 测试函数，验证 bounds 参数包含无限值时的错误
    def test_inf_bounds(self):
        # 定义错误消息
        error_msg = 'Bounds must not be inf.'
        # 创建包含无限值的 bounds 对象
        bounds = Bounds([-np.inf, -1], [-2, np.inf])
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 bounds
            direct(self.styblinski_tang, bounds)

    # 使用 pytest 参数化装饰器标记，验证 locally_biased 参数的类型错误
    @pytest.mark.parametrize("locally_biased", ["bias", [0, 0], 2.])
    def test_locally_biased_validation(self, locally_biased):
        # 定义错误消息
        error_msg = 'locally_biased must be True or False.'
        # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 调用被测函数 direct，传入参数 locally_biased
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   locally_biased=locally_biased)
```