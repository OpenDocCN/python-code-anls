# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_tanhsinh.py`

```
# mypy: disable-error-code="attr-defined"
# 导入标准库和第三方库
import os
import pytest

# 导入 NumPy 库及其子模块
import numpy as np
from numpy.testing import assert_allclose, assert_equal

# 导入 SciPy 的特定模块和函数
import scipy._lib._elementwise_iterative_method as eim
from scipy import special, stats
from scipy.integrate import quad_vec
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache, _nsum
from scipy.stats._discrete_distns import _gen_harmonic_gt1

# 定义测试类 TestTanhSinh
class TestTanhSinh:

    # 定义测试函数 f1，参数为 t
    # 函数返回 t * ln(1 + t)
    def f1(self, t):
        return t * np.log(1 + t)

    # 为函数 f1 添加参考值 ref 和参数 b
    f1.ref = 0.25
    f1.b = 1

    # 类似地定义测试函数 f2 到 f15，每个函数计算不同的数学表达式或函数

    # f2: 返回 t^2 * arctan(t)
    def f2(self, t):
        return t ** 2 * np.arctan(t)

    f2.ref = (np.pi - 2 + 2 * np.log(2)) / 12
    f2.b = 1

    # f3: 返回 exp(t) * cos(t)
    def f3(self, t):
        return np.exp(t) * np.cos(t)

    f3.ref = (np.exp(np.pi / 2) - 1) / 2
    f3.b = np.pi / 2

    # f4: 返回 arctan(sqrt(2 + t^2)) / ((1 + t^2) * sqrt(2 + t^2))
    def f4(self, t):
        a = np.sqrt(2 + t ** 2)
        return np.arctan(a) / ((1 + t ** 2) * a)

    f4.ref = 5 * np.pi ** 2 / 96
    f4.b = 1

    # f5: 返回 sqrt(t) * ln(t)
    def f5(self, t):
        return np.sqrt(t) * np.log(t)

    f5.ref = -4 / 9
    f5.b = 1

    # f6: 返回 sqrt(1 - t^2)
    def f6(self, t):
        return np.sqrt(1 - t ** 2)

    f6.ref = np.pi / 4
    f6.b = 1

    # f7: 返回 sqrt(t) / sqrt(1 - t^2)
    def f7(self, t):
        return np.sqrt(t) / np.sqrt(1 - t ** 2)

    f7.ref = 2 * np.sqrt(np.pi) * special.gamma(3 / 4) / special.gamma(1 / 4)
    f7.b = 1

    # f8: 返回 ln(t)^2
    def f8(self, t):
        return np.log(t) ** 2

    f8.ref = 2
    f8.b = 1

    # f9: 返回 ln(cos(t))
    def f9(self, t):
        return np.log(np.cos(t))

    f9.ref = -np.pi * np.log(2) / 2
    f9.b = np.pi / 2

    # f10: 返回 sqrt(tan(t))
    def f10(self, t):
        return np.sqrt(np.tan(t))

    f10.ref = np.pi * np.sqrt(2) / 2
    f10.b = np.pi / 2

    # f11: 返回 1 / (1 + t^2)
    def f11(self, t):
        return 1 / (1 + t ** 2)

    f11.ref = np.pi / 2
    f11.b = np.inf

    # f12: 返回 exp(-t) / sqrt(t)
    def f12(self, t):
        return np.exp(-t) / np.sqrt(t)

    f12.ref = np.sqrt(np.pi)
    f12.b = np.inf

    # f13: 返回 exp(-t^2 / 2)
    def f13(self, t):
        return np.exp(-t ** 2 / 2)

    f13.ref = np.sqrt(np.pi / 2)
    f13.b = np.inf

    # f14: 返回 exp(-t) * cos(t)
    def f14(self, t):
        return np.exp(-t) * np.cos(t)

    f14.ref = 0.5
    f14.b = np.inf

    # f15: 返回 sin(t) / t
    def f15(self, t):
        return np.sin(t) / t

    f15.ref = np.pi / 2
    f15.b = np.inf

    # 错误计算函数 error，参数为计算结果 res，参考值 ref，是否取对数 log
    def error(self, res, ref, log=False):
        # 计算绝对误差 err
        err = abs(res - ref)

        # 如果不取对数，直接返回绝对误差
        if not log:
            return err

        # 否则，使用 np.errstate 进行错误处理，返回取对数后的误差
        with np.errstate(divide='ignore'):
            return np.log10(err)
    # 定义测试函数 test_input_validation，用于验证输入的有效性
    def test_input_validation(self):
        # 使用 self.f1 作为被测试的函数 f
        f = self.f1

        # 设置错误消息，用于验证 f 必须是可调用的
        message = '`f` must be callable.'
        # 断言调用 _tanhsinh(42, 0, f.b) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(42, 0, f.b)

        # 设置错误消息，用于验证 log 参数必须是 True 或 False
        message = '...must be True or False.'
        # 断言调用 _tanhsinh(f, 0, f.b, log=2) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, log=2)

        # 设置错误消息，用于验证数值必须是实数
        message = '...must be real numbers.'
        # 断言调用 _tanhsinh(f, 1+1j, f.b) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 1+1j, f.b)
        # 断言调用 _tanhsinh(f, 0, f.b, atol='ekki') 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol='ekki')
        # 断言调用 _tanhsinh(f, 0, f.b, rtol=pytest) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=pytest)

        # 设置错误消息，用于验证数值必须是非负有限数
        message = '...must be non-negative and finite.'
        # 断言调用 _tanhsinh(f, 0, f.b, rtol=-1) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=-1)
        # 断言调用 _tanhsinh(f, 0, f.b, atol=np.inf) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf)

        # 设置错误消息，用于验证数值不能是正无穷
        message = '...may not be positive infinity.'
        # 断言调用 _tanhsinh(f, 0, f.b, rtol=np.inf, log=True) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=np.inf, log=True)
        # 断言调用 _tanhsinh(f, 0, f.b, atol=np.inf, log=True) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf, log=True)

        # 设置错误消息，用于验证数值必须是整数
        message = '...must be integers.'
        # 断言调用 _tanhsinh(f, 0, f.b, maxlevel=object()) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=object())
        # 断言调用 _tanhsinh(f, 0, f.b, maxfun=1+1j) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=1+1j)
        # 断言调用 _tanhsinh(f, 0, f.b, minlevel="migratory coconut") 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel="migratory coconut")

        # 设置错误消息，用于验证数值必须是非负数
        message = '...must be non-negative.'
        # 断言调用 _tanhsinh(f, 0, f.b, maxlevel=-1) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=-1)
        # 断言调用 _tanhsinh(f, 0, f.b, maxfun=-1) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=-1)
        # 断言调用 _tanhsinh(f, 0, f.b, minlevel=-1) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel=-1)

        # 设置错误消息，用于验证 preserve_shape 参数必须是 True 或 False
        message = '...must be True or False.'
        # 断言调用 _tanhsinh(f, 0, f.b, preserve_shape=2) 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, preserve_shape=2)

        # 设置错误消息，用于验证 callback 参数必须是可调用的
        message = '...must be callable.'
        # 断言调用 _tanhsinh(f, 0, f.b, callback='elderberry') 时会抛出 ValueError 异常，且异常信息匹配 message
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, callback='elderberry')

    # 使用 pytest.mark.parametrize 标记测试函数，定义不同参数组合的测试案例
    @pytest.mark.parametrize("limits, ref", [
        [(0, np.inf), 0.5],  # b 为正无穷
        [(-np.inf, 0), 0.5],  # a 为负无穷
        [(-np.inf, np.inf), 1],  # a 和 b 均为无穷
        [(np.inf, -np.inf), -1],  # 限制范围颠倒
        [(1, -1), stats.norm.cdf(-1) -  stats.norm.cdf(1)],  # 限制范围颠倒
    ])
    # 测试积分变换函数的正确性，对普通和对数积分都进行检查
    def test_integral_transforms(self, limits, ref):
        # 创建一个标准正态分布对象
        dist = stats.norm()

        # 对 PDF 使用 tanh-sinh 积分变换，并检查积分值是否接近预期值 ref
        res = _tanhsinh(dist.pdf, *limits)
        assert_allclose(res.integral, ref)

        # 对 logPDF 使用 tanh-sinh 积分变换，并检查指数化后的积分值是否接近预期值 ref
        logres = _tanhsinh(dist.logpdf, *limits, log=True)
        assert_allclose(np.exp(logres.integral), ref)
        
        # 检查变换后的积分结果在 ref > 0 时不会不必要地变成复数
        assert (np.issubdtype(logres.integral.dtype, np.floating) if ref > 0
                else np.issubdtype(logres.integral.dtype, np.complexfloating))

        # 检查误差的指数化是否接近原积分结果的误差值
        assert_allclose(np.exp(logres.error), res.error, atol=1e-16)

    # 故意跳过第 15 个测试用例，因为数值上非常困难
    @pytest.mark.parametrize('f_number', range(1, 15))
    def test_basic(self, f_number):
        # 获取对应编号的测试函数 f
        f = getattr(self, f"f{f_number}")
        rtol = 2e-8

        # 使用 tanh-sinh 积分变换计算函数 f 在 [0, f.b] 区间的积分，并检查结果是否在相对误差 rtol 范围内
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert_allclose(res.integral, f.ref, rtol=rtol)

        # 如果 f_number 不是 14，检查相对误差是否小于积分结果的误差
        if f_number not in {14}:  # 在这里略微低估误差
            true_error = abs(self.error(res.integral, f.ref)/res.integral)
            assert true_error < res.error

        # 如果 f_number 是 7、10 或 12，测试成功但未知的情况，则直接返回
        if f_number in {7, 10, 12}:  # 成功，但不知道
            return

        # 断言积分是否成功完成
        assert res.success
        # 断言状态码为 0
        assert res.status == 0

    # 使用不同的参考值 ref 和分布参数 case 测试积分的准确性
    @pytest.mark.parametrize('ref', (0.5, [0.4, 0.6]))
    @pytest.mark.parametrize('case', stats._distr_params.distcont)
    def test_accuracy(self, ref, case):
        distname, params = case

        # 如果分布名称在指定集合中，则跳过，因为 tanh-sinh 对非光滑积分不太适合
        if distname in {'dgamma', 'dweibull', 'laplace', 'kstwo'}:
            pytest.skip('tanh-sinh is not great for non-smooth integrands')

        # 如果分布名称在指定集合中，并且未启用 SCIPY_XSLOW 环境变量，则跳过，因为运行速度太慢
        if (distname in {'studentized_range', 'levy_stable'}
                and not int(os.getenv('SCIPY_XSLOW', 0))):
            pytest.skip('This case passes, but it is too slow.')

        # 创建指定分布名称和参数的分布对象
        dist = getattr(stats, distname)(*params)

        # 计算分布在 ref 区间上的积分，并使用 tanh-sinh 积分变换，检查结果是否接近 ref
        x = dist.interval(ref)
        res = _tanhsinh(dist.pdf, *x)
        assert_allclose(res.integral, ref)

    # 参数化测试形状 shape，用于测试不同维度和尺寸的情况
    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    # 测试向量化函数的功能是否正确，输出形状及数据类型是否符合预期，针对不同的输入形状进行测试。
    def test_vectorization(self, shape):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(82456839535679456794)
        # 使用随机数生成器创建指定形状的随机数组
        a = rng.random(shape)
        b = rng.random(shape)
        p = rng.random(shape)
        # 计算数组元素的乘积
        n = np.prod(shape)

        # 定义一个函数 f，用于对输入数组进行处理，并统计调用次数和评估次数
        def f(x, p):
            f.ncall += 1
            f.feval += 1 if (x.size == n or x.ndim <=1) else x.shape[-1]
            return x**p
        # 初始化函数属性
        f.ncall = 0
        f.feval = 0

        # 使用 numpy.vectorize 装饰器，将函数 _tanhsinh_single 向量化
        @np.vectorize
        def _tanhsinh_single(a, b, p):
            return _tanhsinh(lambda x: x**p, a, b)

        # 调用 _tanhsinh 函数处理数组 a, b，并传递参数 p，返回结果
        res = _tanhsinh(f, a, b, args=(p,))
        # 调用 _tanhsinh_single 函数处理数组 a, b，并展平结果
        refs = _tanhsinh_single(a, b, p).ravel()

        # 检查结果对象的特定属性是否与参考值列表中的相应属性匹配
        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            res_attr = getattr(res, attr)
            # 使用 assert_allclose 检查结果属性值是否接近参考值，相对误差容忍度设置为 1e-15
            assert_allclose(res_attr.ravel(), ref_attr, rtol=1e-15)
            # 检查结果属性的形状是否与输入形状相匹配
            assert_equal(res_attr.shape, shape)

        # 检查 success 属性的数据类型是否为布尔型
        assert np.issubdtype(res.success.dtype, np.bool_)
        # 检查 status 属性的数据类型是否为整数型
        assert np.issubdtype(res.status.dtype, np.integer)
        # 检查 nfev 属性的数据类型是否为整数型
        assert np.issubdtype(res.nfev.dtype, np.integer)
        # 检查 maxlevel 属性的数据类型是否为整数型
        assert np.issubdtype(res.maxlevel.dtype, np.integer)
        # 检查最大 nfev 值是否与函数 f 的评估次数相等
        assert_equal(np.max(res.nfev), f.feval)
        # 检查最大 maxlevel 值是否大于等于 2
        assert np.max(res.maxlevel) >= 2
        # 检查最大 maxlevel 值是否与函数 f 的调用次数相等
        assert_equal(np.max(res.maxlevel), f.ncall)

    # 测试不同状态标志的情况，展示同时产生所有可能状态标志的测试案例。
    def test_flags(self):
        # 定义一个函数 f，对输入数组 xs 进行处理，根据索引数组 js 选择不同的处理函数，并返回结果
        def f(xs, js):
            f.nit += 1
            # 定义三个处理函数，分别表示收敛、达到最大迭代次数和因 NaN 停止
            funcs = [lambda x: np.exp(-x**2),  # 收敛
                     lambda x: np.exp(x),        # 由于 order=2 达到最大迭代次数
                     lambda x: np.full_like(x, np.nan)[()]]  # 因 NaN 停止
            # 对 xs 和 js 中的每个元素应用对应的处理函数，并返回结果列表
            res = [funcs[j](x) for x, j in zip(xs, js.ravel())]
            return res
        # 初始化函数属性 nit
        f.nit = 0

        # 定义参数 args，包含一个整数数组
        args = (np.arange(3, dtype=np.int64),)
        # 调用 _tanhsinh 函数处理数组 [inf, inf, inf] 和 [-inf, -inf, -inf]，并传递参数 args
        res = _tanhsinh(f, [np.inf]*3, [-np.inf]*3, maxlevel=5, args=args)
        # 定义参考标志列表，包含三个整数值，分别表示不同的状态标志
        ref_flags = np.array([0, -2, -3])
        # 使用 assert_equal 检查 res 对象的 status 属性是否与 ref_flags 相等
        assert_equal(res.status, ref_flags)

    # 与上一个测试相同，但使用 preserve_shape 选项简化处理。
    def test_flags_preserve_shape(self):
        # 定义一个函数 f，对输入数组 x 进行处理，根据索引选择不同的处理函数，并返回结果列表
        def f(x):
            return [np.exp(-x[0]**2),  # 收敛
                    np.exp(x[1]),        # 由于 order=2 达到最大迭代次数
                    np.full_like(x[2], np.nan)[()]]  # 因 NaN 停止

        # 调用 _tanhsinh 函数处理数组 [inf, inf, inf] 和 [-inf, -inf, -inf]，并传递参数 preserve_shape=True
        res = _tanhsinh(f, [np.inf]*3, [-np.inf]*3, maxlevel=5, preserve_shape=True)
        # 定义参考标志列表，包含三个整数值，分别表示不同的状态标志
        ref_flags = np.array([0, -2, -3])
        # 使用 assert_equal 检查 res 对象的 status 属性是否与 ref_flags 相等
        assert_equal(res.status, ref_flags)
    def test_preserve_shape(self):
        # Test `preserve_shape` option
        # 定义测试函数 f(x)，返回一个二维数组
        def f(x):
            return np.asarray([[x, np.sin(10 * x)],
                               [np.cos(30 * x), x * np.sin(100 * x)]])

        # 计算参考值 ref，使用 quad_vec 函数对 f 进行数值积分
        ref = quad_vec(f, 0, 1)
        # 调用 _tanhsinh 函数进行积分计算，设置 preserve_shape=True
        res = _tanhsinh(f, 0, 1, preserve_shape=True)
        # 断言计算结果的积分值与参考值的一致性
        assert_allclose(res.integral, ref[0])

    def test_convergence(self):
        # demonstrate that number of accurate digits doubles each iteration
        # 使用 self.f1 作为被积函数 f
        f = self.f1
        # 初始化上一次迭代的误差记录为 0
        last_logerr = 0
        # 进行 4 次迭代
        for i in range(4):
            # 调用 _tanhsinh 函数进行积分计算，设置最小级别为 0，最大级别为 i
            res = _tanhsinh(f, 0, f.b, minlevel=0, maxlevel=i)
            # 计算当前积分误差的对数 logerr
            logerr = self.error(res.integral, f.ref, log=True)
            # 断言误差的对数 logerr 满足精度加倍的关系或小于 -15.5
            assert (logerr < last_logerr * 2 or logerr < -15.5)
            # 更新上一次迭代的误差记录为当前的 logerr
            last_logerr = logerr

    @pytest.mark.parametrize('rtol', [1e-4, 1e-14])
    def test_log(self, rtol):
        # Test equivalence of log-integration and regular integration
        # 创建正态分布对象 dist
        dist = stats.norm()

        # 设置测试的公差参数
        test_tols = dict(atol=1e-18, rtol=1e-15)

        # 正整数被积函数 (真实的对数被积函数)
        # 调用 _tanhsinh 函数计算对数积分，设置 log=True，并指定 rtol 参数
        res = _tanhsinh(dist.logpdf, -1, 2, log=True, rtol=np.log(rtol))
        # 调用 _tanhsinh 函数计算普通积分，指定 rtol 参数
        ref = _tanhsinh(dist.pdf, -1, 2, rtol=rtol)
        # 断言对数积分的结果与普通积分结果的指数函数值近似相等
        assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
        # 断言对数积分的误差与普通积分误差的指数函数值近似相等
        assert_allclose(np.exp(res.error), ref.error, **test_tols)
        # 断言积分计算的函数调用次数相等
        assert res.nfev == ref.nfev

        # 复数被积函数
        def f(x):
            return -dist.logpdf(x)*dist.pdf(x)

        def logf(x):
            return np.log(dist.logpdf(x) + 0j) + dist.logpdf(x) + np.pi * 1j

        # 调用 _tanhsinh 函数计算对数积分，设置 log=True
        res = _tanhsinh(logf, -np.inf, np.inf, log=True)
        # 调用 _tanhsinh 函数计算普通积分
        ref = _tanhsinh(f, -np.inf, np.inf)
        # 在某些 CI 平台上，我们观察到 `invalid` 警告，这里通过设置 np.errstate 忽略所有警告
        with np.errstate(all='ignore'):
            # 断言对数积分的结果与普通积分结果的指数函数值近似相等
            assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
            # 断言对数积分的误差与普通积分误差的指数函数值近似相等
            assert_allclose(np.exp(res.error), ref.error, **test_tols)
        # 断言积分计算的函数调用次数相等
        assert res.nfev == ref.nfev

    def test_complex(self):
        # Test integration of complex integrand
        # 有限积分区间下的复数被积函数
        def f(x):
            return np.exp(1j * x)

        # 调用 _tanhsinh 函数计算复数被积函数的积分，积分区间为 [0, π/4]
        res = _tanhsinh(f, 0, np.pi/4)
        # 参考值 ref
        ref = np.sqrt(2)/2 + (1-np.sqrt(2)/2)*1j
        # 断言计算结果的积分值与参考值的一致性
        assert_allclose(res.integral, ref)

        # 无限积分区间下的复数被积函数
        dist1 = stats.norm(scale=1)
        dist2 = stats.norm(scale=2)
        def f(x):
            return dist1.pdf(x) + 1j*dist2.pdf(x)

        # 调用 _tanhsinh 函数计算复数被积函数的积分，积分区间为 [-∞, +∞]
        res = _tanhsinh(f, np.inf, -np.inf)
        # 参考值 ref
        assert_allclose(res.integral, -(1+1j))

    @pytest.mark.parametrize("maxlevel", range(4))
    # 定义一个测试函数，用于验证在给定的最大级别下，最小级别对被积函数的评估值、积分/误差估计值没有改变，只影响函数调用次数
    def test_minlevel(self, maxlevel):
        # 定义被积函数f(x)，记录函数调用次数、评估数量以及所有评估点的数组
        def f(x):
            f.calls += 1  # 增加函数调用次数计数
            f.feval += np.size(x)  # 记录评估数量
            f.x = np.concatenate((f.x, x.ravel()))  # 将当前评估点添加到所有评估点的数组中
            return self.f2(x)  # 返回被积函数self.f2(x)
        f.feval, f.calls, f.x = 0, 0, np.array([])  # 初始化评估数量、函数调用次数和评估点数组

        # 使用_tanhsinh函数计算积分，设置最小级别为0，最大级别为maxlevel
        ref = _tanhsinh(f, 0, self.f2.b, minlevel=0, maxlevel=maxlevel)
        ref_x = np.sort(f.x)  # 对所有评估点的数组进行排序

        # 遍历最小级别从0到maxlevel的所有情况
        for minlevel in range(0, maxlevel + 1):
            f.feval, f.calls, f.x = 0, 0, np.array([])  # 重置评估数量、函数调用次数和评估点数组
            options = dict(minlevel=minlevel, maxlevel=maxlevel)  # 设置选项字典，包括最小和最大级别
            res = _tanhsinh(f, 0, self.f2.b, **options)  # 使用_tanhsinh函数计算积分
            # 应该非常接近；变化仅仅是评估点的顺序
            assert_allclose(res.integral, ref.integral, rtol=4e-16)
            # 绝对误差的差异 << 积分的数量级
            assert_allclose(res.error, ref.error, atol=4e-16 * ref.integral)
            assert res.nfev == f.feval == len(f.x)  # 检查函数调用次数与评估数量是否匹配
            assert f.calls == maxlevel - minlevel + 1 + 1  # 1个验证调用
            assert res.status == ref.status  # 检查结果状态是否匹配
            assert_equal(ref_x, np.sort(f.x))  # 检查评估点数组是否与参考值ref_x排序后相同

    # 测试处理积分上限为无穷大的情况（与有限上限混合）
    def test_improper_integrals(self):
        # 定义被积函数f(x)，将无穷大的值替换为NaN，返回指数函数的值
        def f(x):
            x[np.isinf(x)] = np.nan
            return np.exp(-x**2)
        a = [-np.inf, 0, -np.inf, np.inf, -20, -np.inf, -20]  # 上限数组
        b = [np.inf, np.inf, 0, -np.inf, 20, 20, np.inf]  # 下限数组
        ref = np.sqrt(np.pi)  # 参考值为根号π
        res = _tanhsinh(f, a, b)  # 使用_tanhsinh函数计算积分
        # 检查积分结果是否非常接近参考值
        assert_allclose(res.integral, [ref, ref/2, ref/2, -ref, ref, ref, ref])

    # 参数化测试，测试数据类型是否保持不变
    @pytest.mark.parametrize("limits", ((0, 3), ([-np.inf, 0], [3, 3])))
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_dtype(self, limits, dtype):
        # 测试数据类型是否保持不变
        a, b = np.asarray(limits, dtype=dtype)[()]

        # 定义被积函数f(x)，确保输入x的数据类型与dtype一致，返回指数函数的值
        def f(x):
            assert x.dtype == dtype  # 检查输入x的数据类型是否为dtype
            return np.exp(x)

        rtol = 1e-12 if dtype == np.float64 else 1e-5  # 设置相对误差tolerance
        res = _tanhsinh(f, a, b, rtol=rtol)  # 使用_tanhsinh函数计算积分
        assert res.integral.dtype == dtype  # 检查积分结果的数据类型是否为dtype
        assert res.error.dtype == dtype  # 检查误差估计的数据类型是否为dtype
        assert np.all(res.success)  # 检查所有积分是否成功
        assert_allclose(res.integral, np.exp(b)-np.exp(a), rtol=rtol)  # 检查积分结果是否非常接近预期值
    def test_maxiter_callback(self):
        # 测试 `maxiter` 参数和 `callback` 接口的行为

        a, b = -np.inf, np.inf
        def f(x):
            return np.exp(-x*x)

        minlevel, maxlevel = 0, 2
        maxiter = maxlevel - minlevel + 1
        kwargs = dict(minlevel=minlevel, maxlevel=maxlevel, rtol=1e-15)

        # 调用 _tanhsinh 函数进行数值积分计算
        res = _tanhsinh(f, a, b, **kwargs)

        # 断言结果中 success 属性为 False
        assert not res.success
        # 断言结果中 maxlevel 属性与设定的 maxlevel 相等
        assert res.maxlevel == maxlevel

        # 定义一个回调函数 callback
        def callback(res):
            # 每调用一次 callback，iter 计数加一
            callback.iter += 1
            # 记录当前的 res
            callback.res = res
            # 断言 res 对象中包含 integral 属性
            assert hasattr(res, 'integral')
            # 断言 res 对象的 status 属性为 1
            assert res.status == 1
            # 如果 iter 达到 maxiter，则抛出 StopIteration 异常
            if callback.iter == maxiter:
                raise StopIteration

        # 初始化 callback 的计数和 res
        callback.iter = -1  # 在第一次迭代之前调用一次 callback
        callback.res = None

        # 删除 kwargs 中的 maxlevel 参数
        del kwargs['maxlevel']

        # 调用 _tanhsinh 函数，传入 callback 函数作为参数
        res2 = _tanhsinh(f, a, b, **kwargs, callback=callback)

        # 用 callback 终止的结果应与 maxiter 终止的结果相同（除了 status 属性）
        for key in res.keys():
            if key == 'status':
                # 断言 callback 返回的 res 的 status 属性为 1
                assert callback.res[key] == 1
                # 断言 res 和 res2 的 status 属性分别为 -2 和 -4
                assert res[key] == -2
                assert res2[key] == -4
            else:
                # 断言 res2、callback.res 和 res 的其它属性值相同
                assert res2[key] == callback.res[key] == res[key]

    def test_jumpstart(self):
        # 在每个级别 i 处的中间结果应与从级别 i 起跳的最终结果相同；
        # 即 minlevel=maxlevel=i

        a, b = -np.inf, np.inf
        def f(x):
            return np.exp(-x*x)

        # 定义一个 callback 函数，用于记录积分和误差
        def callback(res):
            callback.integrals.append(res.integral)
            callback.errors.append(res.error)
        callback.integrals = []
        callback.errors = []

        maxlevel = 4

        # 第一次调用 _tanhsinh 函数，设置 minlevel=0, maxlevel=maxlevel，并传入 callback
        _tanhsinh(f, a, b, minlevel=0, maxlevel=maxlevel, callback=callback)

        # 分别计算 minlevel=i, maxlevel=i 的积分和误差
        integrals = []
        errors = []
        for i in range(maxlevel + 1):
            res = _tanhsinh(f, a, b, minlevel=i, maxlevel=i)
            integrals.append(res.integral)
            errors.append(res.error)

        # 断言 callback 记录的积分结果与直接计算的结果相近
        assert_allclose(callback.integrals[1:], integrals, rtol=1e-15)
        # 断言 callback 记录的误差结果与直接计算的结果相近，同时考虑到绝对误差
        assert_allclose(callback.errors[1:], errors, rtol=1e-15, atol=1e-16)
    def test_special_cases(self):
        # 测试边界情况和其他特殊情况

        # 定义一个函数 `f`，检查传入的参数是否为浮点数类型
        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99

        # 调用 `_tanhsinh` 函数计算积分，范围是 [0, 1]
        res = _tanhsinh(f, 0, 1)
        # 断言计算成功
        assert res.success
        # 断言积分结果接近于 1/100
        assert_allclose(res.integral, 1/100)

        # 使用 `maxlevel=0` 测试级别 0 的情况，此时错误应为 NaN
        res = _tanhsinh(f, 0, 1, maxlevel=0)
        assert res.integral > 0  # 确保积分结果大于0
        assert_equal(res.error, np.nan)  # 确保错误值为 NaN
        # 使用 `maxlevel=1` 测试级别 1 的情况，同样错误应为 NaN
        res = _tanhsinh(f, 0, 1, maxlevel=1)
        assert res.integral > 0
        assert_equal(res.error, np.nan)

        # 测试左右积分限相等的情况
        res = _tanhsinh(f, 1, 1)
        assert res.success
        assert res.maxlevel == -1
        assert_allclose(res.integral, 0)

        # 定义一个带两个参数的函数 `f`，其中第二个参数是指数
        def f(x, c):
            return x**c

        # 测试传入标量参数 `args=99` 的情况
        res = _tanhsinh(f, 0, 1, args=99)
        assert_allclose(res.integral, 1/100)

        # 测试包含 NaN 值的情况
        a = [np.nan, 0, 0, 0]
        b = [1, np.nan, 1, 1]
        c = [1, 1, np.nan, 1]
        res = _tanhsinh(f, a, b, args=(c,))
        assert_allclose(res.integral, [np.nan, np.nan, np.nan, 0.5])
        assert_allclose(res.error[:3], np.nan)
        assert_equal(res.status, [-3, -3, -3, 0])
        assert_equal(res.success, [False, False, False, True])
        assert_equal(res.nfev[:3], 1)

        # 测试复合积分后跟实数积分的情况
        # 确保在不同类型积分之间转换时避免复杂数警告
        _pair_cache.xjc = np.empty(0)
        _pair_cache.wj = np.empty(0)
        _pair_cache.indices = [0]
        _pair_cache.h0 = None
        res = _tanhsinh(lambda x: x*1j, 0, 1)
        assert_allclose(res.integral, 0.5*1j)
        res = _tanhsinh(lambda x: x, 0, 1)
        assert_allclose(res.integral, 0.5)

        # 测试零尺寸输入的情况
        shape = (0, 3)
        res = _tanhsinh(lambda x: x, 0, np.zeros(shape))
        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            assert_equal(res[attr].shape, shape)
class TestNSum:
    # 使用固定种子创建一个随机数生成器对象
    rng = np.random.default_rng(5895448232066142650)
    # 从均匀分布中生成大小为10的随机数数组
    p = rng.uniform(1, 10, size=10)

    # 计算 k 的负二次幂，用于函数 f1
    def f1(self, k):
        # Integers are never passed to `f1`; if they were, we'd get
        # integer to negative integer power error
        return k**(-2)

    # 将参考值 np.pi**2/6 赋给 f1 函数的属性 ref
    f1.ref = np.pi**2/6
    # 设置 f1 函数的属性 a 和 b
    f1.a = 1
    f1.b = np.inf
    # 空元组作为 f1 函数的参数列表
    f1.args = tuple()

    # 计算函数 f2，返回 1/k**p
    def f2(self, k, p):
        return 1 / k**p

    # 使用特殊函数库计算 zeta 函数值，并将其赋给 f2 函数的属性 ref
    f2.ref = special.zeta(p, 1)
    # 设置 f2 函数的属性 a 和 b
    f2.a = 1
    f2.b = np.inf
    # 将 p 作为单元素元组赋给 f2 函数的参数列表
    f2.args = (p,)

    # 计算函数 f3，返回 1/k**p
    def f3(self, k, p):
        return 1 / k**p

    # 设置 f3 函数的属性 a
    f3.a = 1
    # 从指定范围生成随机整数数组，并传递给 _gen_harmonic_gt1 函数，将结果赋给 f3 函数的属性 ref
    f3.b = rng.integers(5, 15, size=(3, 1))
    f3.ref = _gen_harmonic_gt1(f3.b, p)
    # 将 p 作为单元素元组赋给 f3 函数的参数列表
    f3.args = (p,)

    # 测试输入验证功能
    def test_input_validation(self):
        # 选择 f1 函数作为被测试的函数
        f = self.f1

        # 定义错误消息
        message = '`f` must be callable.'
        # 使用 pytest 的断言检查 ValueError 异常是否被正确抛出，并匹配错误消息
        with pytest.raises(ValueError, match=message):
            _nsum(42, f.a, f.b)

        message = '...must be True or False.'
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, log=2)

        message = '...must be real numbers.'
        with pytest.raises(ValueError, match=message):
            _nsum(f, 1+1j, f.b)
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, None)
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, step=object())
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, atol='ekki')
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, rtol=pytest)

        # 忽略所有 numpy 的运行时错误
        with np.errstate(all='ignore'):
            # 测试包含 NaN、-Inf 和 Inf 的情况，以及检查结果对象的属性值
            res = _nsum(f, [np.nan, -np.inf, np.inf], 1)
            assert np.all((res.status == -1) & np.isnan(res.sum)
                          & np.isnan(res.error) & ~res.success & res.nfev == 1)
            res = _nsum(f, 10, [np.nan, 1])
            assert np.all((res.status == -1) & np.isnan(res.sum)
                          & np.isnan(res.error) & ~res.success & res.nfev == 1)
            res = _nsum(f, 1, 10, step=[np.nan, -np.inf, np.inf, -1, 0])
            assert np.all((res.status == -1) & np.isnan(res.sum)
                          & np.isnan(res.error) & ~res.success & res.nfev == 1)

        message = '...must be non-negative and finite.'
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, rtol=-1)
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, atol=np.inf)

        message = '...may not be positive infinity.'
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, rtol=np.inf, log=True)
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, atol=np.inf, log=True)

        message = '...must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, maxterms=3.5)
        with pytest.raises(ValueError, match=message):
            _nsum(f, f.a, f.b, maxterms=-2)

    # 使用 pytest 的参数化装饰器，设置参数 f_number 的范围为 1 到 4
    @pytest.mark.parametrize('f_number', range(1, 4))
    # 定义用于测试基本情况的方法，接受一个测试编号作为参数
    def test_basic(self, f_number):
        # 获取测试对象 f，通过反射机制获取对象的方法或属性
        f = getattr(self, f"f{f_number}")
        # 调用 _nsum 函数计算结果，传入函数 f 的参数和其他参数
        res = _nsum(f, f.a, f.b, args=f.args)
        # 断言计算结果的和与参考值相近
        assert_allclose(res.sum, f.ref)
        # 断言计算结果的状态为 0
        assert_equal(res.status, 0)
        # 断言计算成功
        assert_equal(res.success, True)

        # 忽略除法相关的数值错误
        with np.errstate(divide='ignore'):
            # 计算对数和，使用 lambda 表达式调用 f 函数
            logres = _nsum(lambda *args: np.log(f(*args)),
                           f.a, f.b, log=True, args=f.args)
        # 断言对数和的指数与原始结果的和相近
        assert_allclose(np.exp(logres.sum), res.sum)
        # 断言对数和的误差的指数与原始结果的误差相近
        assert_allclose(np.exp(logres.error), res.error)
        # 断言对数和的状态为 0
        assert_equal(logres.status, 0)
        # 断言对数和的计算成功
        assert_equal(logres.success, True)

    @pytest.mark.parametrize('maxterms', [0, 1, 10, 20, 100])
    # 使用参数化测试，测试积分方法，maxterms 为参数
    def test_integral(self, maxterms):
        # 使用第一个测试函数 f1 作为测试对象 f
        f = self.f1

        # 定义对数函数 logf 和积分函数 F
        def logf(x):
            return -2*np.log(x)

        def F(x):
            return -1 / x

        # 初始化积分的起始点 a 和结束点 b，及步长 step
        a = np.asarray([1, 5])[:, np.newaxis]
        b = np.asarray([20, 100, np.inf])[:, np.newaxis, np.newaxis]
        step = np.asarray([0.5, 1, 2]).reshape((-1, 1, 1, 1))
        # 计算步数 nsteps
        nsteps = np.floor((b - a)/step)
        # 保存原始结束点 b
        b_original = b
        # 调整结束点 b 为整数步长
        b = a + nsteps*step

        # 计算部分和的结束点 k
        k = a + maxterms*step
        # 计算部分和 direct
        direct = f(a + np.arange(maxterms)*step).sum(axis=-1, keepdims=True)
        # 计算积分估计值 integral，用于估计剩余部分的积分
        integral = (F(b) - F(k))/step
        # 计算理论下界 low 和上界 high
        low = direct + integral + f(b)
        high = direct + integral + f(k)
        # 计算参考和 ref_sum，_nsum 使用两者的平均值
        ref_sum = (low + high)/2
        # 计算误差 ref_err，假设完美的数值积分
        ref_err = (high - low)/2

        # 当项数小于 maxterms 时，修正参考值
        a, b, step = np.broadcast_arrays(a, b, step)
        for i in np.ndindex(a.shape):
            ai, bi, stepi = a[i], b[i], step[i]
            if (bi - ai)/stepi + 1 <= maxterms:
                direct = f(np.arange(ai, bi+stepi, stepi)).sum()
                ref_sum[i] = direct
                ref_err[i] = direct * np.finfo(direct).eps

        # 设置相对误差容差 rtol
        rtol = 1e-12
        # 调用 _nsum 计算积分结果 res
        res = _nsum(f, a, b_original, step=step, maxterms=maxterms, rtol=rtol)
        # 断言计算结果的和与参考和相近
        assert_allclose(res.sum, ref_sum, rtol=10*rtol)
        # 断言计算结果的误差与参考误差相近
        assert_allclose(res.error, ref_err, rtol=100*rtol)
        # 断言计算结果的状态为 0
        assert_equal(res.status, 0)
        # 断言计算成功
        assert_equal(res.success, True)

        # 选择项数小于 maxterms 的结果进行验证
        i = ((b_original - a)/step + 1 <= maxterms)
        # 断言部分和与参考和相近
        assert_allclose(res.sum[i], ref_sum[i], rtol=1e-15)
        # 断言误差与参考误差相近
        assert_allclose(res.error[i], ref_err[i], rtol=1e-15)

        # 计算对数和 logres
        logres = _nsum(logf, a, b_original, step=step, log=True,
                       rtol=np.log(rtol), maxterms=maxterms)
        # 断言对数和的指数和与原始结果的和相近
        assert_allclose(np.exp(logres.sum), res.sum)
        # 断言对数和的误差的指数和与原始结果的误差相近
        assert_allclose(np.exp(logres.error), res.error)
        # 断言对数和的状态为 0
        assert_equal(logres.status, 0)
        # 断言对数和的计算成功
        assert_equal(logres.success, True)
    def test_vectorization(self, shape):
        # 测试向量化功能的正确性、输出形状和数据类型，针对不同的输入形状。
        rng = np.random.default_rng(82456839535679456794)
        # 使用指定种子生成随机数生成器
        a = rng.integers(1, 10, size=shape)
        # 生成指定形状的随机整数数组a

        # 当和可以直接计算或`maxterms`足够大以满足`atol`时，
        # 向量化调用和循环之间可能会有轻微差异（有充分理由）。
        b = np.inf
        # 设置b为无穷大
        p = rng.random(shape) + 1
        # 生成具有指定形状的随机浮点数数组p，范围为(1, 2)

        n = np.prod(shape)
        # 计算形状shape中元素数量的乘积

        def f(x, p):
            # 定义一个函数f，用于计算特定表达式
            f.feval += 1 if (x.size == n or x.ndim <= 1) else x.shape[-1]
            # 根据x的大小或维度来更新feval计数器
            return 1 / x ** p

        f.feval = 0
        # 初始化feval计数器为0

        @np.vectorize
        def _nsum_single(a, b, p, maxterms):
            # 使用np.vectorize装饰器，向量化地计算_nsum_single函数
            return _nsum(lambda x: 1 / x**p, a, b, maxterms=maxterms)

        # 调用_nsum_single函数，返回结果并展平
        res = _nsum(f, a, b, maxterms=1000, args=(p,))
        # 调用_nsum函数，使用f函数对输入a, b进行求和，设置最大项数为1000
        refs = _nsum_single(a, b, p, maxterms=1000).ravel()
        # 调用_nsum_single函数，使用a, b, p进行向量化求和，展平结果

        attrs = ['sum', 'error', 'success', 'status', 'nfev']
        # 定义属性列表
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            # 获取refs中每个元素的指定属性值列表
            res_attr = getattr(res, attr)
            # 获取res对象的指定属性值
            assert_allclose(res_attr.ravel(), ref_attr, rtol=1e-15)
            # 断言res_attr与ref_attr的值在指定的相对误差范围内相等
            assert_equal(res_attr.shape, shape)
            # 断言res_attr的形状与shape相等

        assert np.issubdtype(res.success.dtype, np.bool_)
        # 断言res.success的数据类型是布尔类型的子类型
        assert np.issubdtype(res.status.dtype, np.integer)
        # 断言res.status的数据类型是整数类型的子类型
        assert np.issubdtype(res.nfev.dtype, np.integer)
        # 断言res.nfev的数据类型是整数类型的子类型
        assert_equal(np.max(res.nfev), f.feval)
        # 断言res.nfev的最大值等于feval计数器的值

    def test_status(self):
        f = self.f2
        # 将self.f2赋值给f

        p = [2, 2, 0.9, 1.1]
        # 定义列表p
        a = [0, 0, 1, 1]
        # 定义列表a
        b = [10, np.inf, np.inf, np.inf]
        # 定义列表b
        ref = special.zeta(p, 1)
        # 计算特殊函数zeta的值，赋值给ref

        with np.errstate(divide='ignore'):  # 有意除以零
            # 忽略除法错误的上下文管理器
            res = _nsum(f, a, b, args=(p,))
            # 调用_nsum函数，使用f函数对输入a, b进行求和，设置附加参数为p

        assert_equal(res.success, [False, False, False, True])
        # 断言res.success与指定的列表值相等
        assert_equal(res.status, [-3, -3, -2, 0])
        # 断言res.status与指定的列表值相等
        assert_allclose(res.sum[res.success], ref[res.success])
        # 断言在res.success为True的索引处，res.sum与ref的值在指定的相对误差范围内相等

    def test_nfev(self):
        def f(x):
            f.nfev += np.size(x)
            # 更新nfev计数器的值
            return 1 / x**2

        f.nfev = 0
        # 初始化nfev计数器为0
        res = _nsum(f, 1, 10)
        # 调用_nsum函数，使用f函数对输入1到10进行求和
        assert_equal(res.nfev, f.nfev)
        # 断言res.nfev与nfev计数器的值相等

        f.nfev = 0
        # 重置nfev计数器为0
        res = _nsum(f, 1, np.inf, atol=1e-6)
        # 调用_nsum函数，使用f函数对输入1到无穷大进行求和，设置绝对误差限制
        assert_equal(res.nfev, f.nfev)
        # 断言res.nfev与nfev计数器的值相等

    def test_inclusive(self):
        # 当调用_direct时，当inclusive=True时有一个边界情况偏差问题。
        # 确保这个问题已经解决。
        res = _nsum(lambda k: 1 / k ** 2, [1, 4], np.inf, maxterms=500, atol=0.1)
        # 调用_nsum函数，使用lambda函数对输入[1, 4], np.inf进行求和，设置最大项数和绝对误差限制
        ref = _nsum(lambda k: 1 / k ** 2, [1, 4], np.inf)
        # 调用_nsum函数，使用lambda函数对输入[1, 4], np.inf进行求和
        assert np.all(res.sum > (ref.sum - res.error))
        # 断言res.sum大于(ref.sum - res.error)中的所有元素
        assert np.all(res.sum < (ref.sum + res.error))
        # 断言res.sum小于(ref.sum + res.error)中的所有元素
    def test_special_case(self):
        # 测试特殊情况：相等的上下限
        f = self.f1
        a = b = 2
        # 调用 _nsum 函数计算结果
        res = _nsum(f, a, b)
        # 断言结果的和与 f(a) 相等
        assert_equal(res.sum, f(a))

        # 测试标量 `args`（不在元组中）
        res = _nsum(self.f2, 1, np.inf, args=2)
        # 断言结果的和与 self.f1.ref 相近，使用 args=2 时的正确值
        assert_allclose(res.sum, self.f1.ref)

        # 测试输入为零大小的情况
        a = np.empty((3, 1, 1))  # 任意广播形状
        b = np.empty((0, 1))  # 可能使用 Hypothesis
        p = np.empty(4)  # 但这有些过头了
        # 计算广播后的形状
        shape = np.broadcast_shapes(a.shape, b.shape, p.shape)
        # 调用 _nsum 函数计算结果
        res = _nsum(self.f2, a, b, args=(p,))
        # 断言结果的形状与预期的广播后形状相等
        assert res.sum.shape == shape
        assert res.status.shape == shape
        assert res.nfev.shape == shape

        # 测试 maxterms=0 的情况
        def f(x):
            with np.errstate(divide='ignore'):
                return 1 / x

        # 调用 _nsum 函数计算结果
        res = _nsum(f, 0, 10, maxterms=0)
        # 断言结果的和为 NaN
        assert np.isnan(res.sum)
        assert np.isnan(res.error)
        # 断言结果的状态为 -2
        assert res.status == -2

        # 调用 _nsum 函数计算结果
        res = _nsum(f, 0, 10, maxterms=1)
        # 断言结果的和为 NaN
        assert np.isnan(res.sum)
        assert np.isnan(res.error)
        # 断言结果的状态为 -3
        assert res.status == -3

        # 测试 NaN 的情况
        # 如果存在 NaN，应该跳过直接和积分方法
        a = [np.nan, 1, 1, 1]
        b = [np.inf, np.nan, np.inf, np.inf]
        p = [2, 2, np.nan, 2]
        # 调用 _nsum 函数计算结果
        res = _nsum(self.f2, a, b, args=(p,))
        # 断言结果的和应与预期的数组匹配（包含 NaN）
        assert_allclose(res.sum, [np.nan, np.nan, np.nan, self.f1.ref])
        # 断言前三个元素的误差都为 NaN
        assert_allclose(res.error[:3], np.nan)
        # 断言结果的状态符合预期
        assert_equal(res.status, [-1, -1, -3, 0])
        # 断言结果的成功状态符合预期
        assert_equal(res.success, [False, False, False, True])
        # 断言前两个元素的函数评估次数为 1
        assert_equal(res.nfev[:2], 1)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_dtype(self, dtype):
        def f(k):
            assert k.dtype == dtype
            return 1 / k ** np.asarray(2, dtype=dtype)[()]

        a = np.asarray(1, dtype=dtype)
        b = np.asarray([10, np.inf], dtype=dtype)
        # 调用 _nsum 函数计算结果
        res = _nsum(f, a, b)
        # 断言结果的和与预期值相等
        assert res.sum.dtype == dtype
        assert res.error.dtype == dtype

        # 根据 dtype 设置相对误差的阈值
        rtol = 1e-12 if dtype == np.float64 else 1e-6
        # 生成大于 1 的调和级数作为参考值
        ref = _gen_harmonic_gt1(b, 2)
        # 断言结果的和与预期的参考值相近
        assert_allclose(res.sum, ref, rtol=rtol)
```