# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_differentiate.py`

```
import pytest  # 导入 pytest 模块

import numpy as np  # 导入 NumPy 库，命名为 np
from numpy.testing import assert_allclose  # 导入 NumPy 测试模块中的 assert_allclose 函数

from scipy.conftest import array_api_compatible  # 导入 scipy 的 conftest 模块中的 array_api_compatible
import scipy._lib._elementwise_iterative_method as eim  # 导入 scipy._lib._elementwise_iterative_method 模块，命名为 eim
from scipy._lib._array_api import (xp_assert_close, xp_assert_equal, xp_assert_less,  # 导入 scipy._lib._array_api 中的多个函数
                                   is_numpy, is_torch, array_namespace)

from scipy import stats, optimize, special  # 导入 scipy 中的 stats, optimize, special 模块
from scipy.optimize._differentiate import (_differentiate as differentiate,  # 导入 scipy.optimize._differentiate 模块中的函数
                                           _jacobian as jacobian, _EERRORINCREASE)

@array_api_compatible  # 使用 array API 兼容性修饰器
@pytest.mark.usefixtures("skip_xp_backends")  # 使用 pytest 的 usefixtures 修饰器，跳过指定的 XP 后端
@pytest.mark.skip_xp_backends('array_api_strict', 'jax.numpy',  # 使用 pytest 的 skip_xp_backends 修饰器，跳过指定的 XP 后端
                              reasons=['Currently uses fancy indexing assignment.',  # 给出跳过原因的列表
                                       'JAX arrays do not support item assignment.'])
class TestDifferentiate:  # 定义测试类 TestDifferentiate

    def f(self, x):  # 定义实例方法 f，接受参数 x
        return special.ndtr(x)  # 调用 special 模块中的 ndtr 函数，返回结果

    @pytest.mark.parametrize('x', [0.6, np.linspace(-0.05, 1.05, 10)])  # 使用 pytest 的 parametrize 修饰器，参数化测试方法的参数 x
    def test_basic(self, x, xp):  # 定义测试方法 test_basic，接受参数 x 和 xp
        # Invert distribution CDF and compare against distribution `ppf`
        default_dtype = xp.asarray(1.).dtype  # 使用 xp 的 asarray 函数创建数组，并获取其数据类型
        res = differentiate(self.f, xp.asarray(x, dtype=default_dtype))  # 调用 differentiate 函数，对 self.f 在 x 处进行数值微分，结果存储在 res 中
        ref = xp.asarray(stats.norm().pdf(x), dtype=default_dtype)  # 使用 xp 计算标准正态分布在 x 处的概率密度函数值，并将结果存储在 ref 中
        xp_assert_close(res.df, ref)  # 使用 xp 的 xp_assert_close 函数断言 res.df 与 ref 接近
        # This would be nice, but doesn't always work out. `error` is an
        # estimate, not a bound.
        if not is_torch(xp):  # 如果 xp 不是 torch
            xp_assert_less(xp.abs(res.df - ref), res.error)  # 使用 xp 的 xp_assert_less 函数断言 res.df 与 ref 之差的绝对值小于 res.error

    @pytest.mark.skip_xp_backends(np_only=True)  # 使用 pytest 的 skip_xp_backends 修饰器，仅跳过 NumPy 后端
    @pytest.mark.parametrize('case', stats._distr_params.distcont)  # 使用 pytest 的 parametrize 修饰器，参数化测试方法的参数 case
    def test_accuracy(self, case):  # 定义测试方法 test_accuracy，接受参数 case
        distname, params = case  # 解包 case 元组，获取分布名称和参数
        dist = getattr(stats, distname)(*params)  # 根据 distname 获取 stats 模块中的分布对象，并使用 params 实例化该分布
        x = dist.median() + 0.1  # 计算该分布的中位数，并加上 0.1
        res = differentiate(dist.cdf, x)  # 对 dist.cdf 在 x 处进行数值微分，结果存储在 res 中
        ref = dist.pdf(x)  # 计算该分布在 x 处的概率密度函数值，并将结果存储在 ref 中
        assert_allclose(res.df, ref, atol=1e-10)  # 使用 assert_allclose 函数断言 res.df 与 ref 接近，容差为 1e-10

    @pytest.mark.parametrize('order', [1, 6])  # 使用 pytest 的 parametrize 修饰器，参数化测试方法的参数 order
    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])  # 使用 pytest 的 parametrize 修饰器，参数化测试方法的参数 shape
    # 定义一个测试向量化功能的方法，用于测试不同输入形状的正确功能、输出形状和数据类型。
    def test_vectorization(self, order, shape, xp):
        # 如果指定了形状，则生成一个指定形状的线性空间数据；否则使用默认值 0.6。
        x = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        # 计算数组 x 的元素个数
        n = np.size(x)

        # 使用 np.vectorize 装饰器定义一个向量化的函数 _differentiate_single
        @np.vectorize
        def _differentiate_single(x):
            # 调用 differentiate 函数对 self.f 在 x 处进行 order 阶的求导计算
            return differentiate(self.f, x, order=order)

        # 定义函数 f，增加计数器 nit 和 feval，根据输入 x 的形状调整 feval 的计数方式
        def f(x, *args, **kwargs):
            f.nit += 1
            f.feval += 1 if (x.size == n or x.ndim <=1) else x.shape[-1]
            return self.f(x, *args, **kwargs)
        # 初始化计数器
        f.nit = -1
        f.feval = 0

        # 调用 differentiate 函数对 x 进行 order 阶的求导计算，使用 xp.asarray 将 x 转换为 xp.float64 类型
        res = differentiate(f, xp.asarray(x, dtype=xp.float64), order=order)
        # 调用 _differentiate_single 对 x 进行求导，将结果展平为一维数组
        refs = _differentiate_single(x).ravel()

        # 提取 refs 中的 x 值，并使用 xp_assert_close 检查与 res.x 的近似性
        ref_x = [ref.x for ref in refs]
        xp_assert_close(xp.reshape(res.x, (-1,)), xp.asarray(ref_x))

        # 提取 refs 中的 df 值，并使用 xp_assert_close 检查与 res.df 的近似性
        ref_df = [ref.df for ref in refs]
        xp_assert_close(xp.reshape(res.df, (-1,)), xp.asarray(ref_df))

        # 提取 refs 中的 error 值，并使用 xp_assert_close 检查与 res.error 的近似性，设置容差为 1e-12
        ref_error = [ref.error for ref in refs]
        xp_assert_close(xp.reshape(res.error, (-1,)), xp.asarray(ref_error),
                        atol=1e-12)

        # 提取 refs 中的 success 值，并使用 xp_assert_equal 检查与 res.success 的相等性
        ref_success = [bool(ref.success) for ref in refs]
        xp_assert_equal(xp.reshape(res.success, (-1,)), xp.asarray(ref_success))

        # 提取 refs 中的 status 值，并使用 xp_assert_equal 检查与 res.status 的相等性
        ref_flag = [np.int32(ref.status) for ref in refs]
        xp_assert_equal(xp.reshape(res.status, (-1,)), xp.asarray(ref_flag))

        # 提取 refs 中的 nfev 值，并使用 xp_assert_equal 检查与 res.nfev 的相等性
        ref_nfev = [np.int32(ref.nfev) for ref in refs]
        xp_assert_equal(xp.reshape(res.nfev, (-1,)), xp.asarray(ref_nfev))
        # 如果 xp 不是 numpy，则无法预期其他后端与 f.feval 完全相同
        if not is_numpy(xp):
            xp.max(res.nfev) == f.feval

        # 提取 refs 中的 nit 值，并使用 xp_assert_equal 检查与 res.nit 的相等性
        ref_nit = [np.int32(ref.nit) for ref in refs]
        # 如果 xp 不是 numpy，则无法预期其他后端与 f.nit 完全相同
        xp_assert_equal(xp.reshape(res.nit, (-1,)), xp.asarray(ref_nit))
        if not is_numpy(xp):
            xp.max(res.nit) == f.nit

    # 定义一个测试不同状态标志的方法，展示同时生成所有可能的状态标志。
    def test_flags(self, xp):
        # 使用指定的随机数生成器 rng 定义函数 f，增加计数器 nit，并根据输入 xs 和 js 生成结果
        rng = np.random.default_rng(5651219684984213)
        def f(xs, js):
            f.nit += 1
            # 定义多个函数并在给定的 x 上进行计算，生成结果列表 res
            funcs = [lambda x: x - 2.5,  # 收敛
                     lambda x: xp.exp(x)*rng.random(),  # 错误增加
                     lambda x: xp.exp(x),  # 由于 order=2 达到最大迭代次数
                     lambda x: xp.full_like(x, xp.nan)[()]]  # 因 NaN 停止
            res = [funcs[int(j)](x) for x, j in zip(xs, xp.reshape(js, (-1,)))]
            return xp.stack(res)
        # 初始化计数器
        f.nit = 0

        # 定义参数 args 为 xp.arange(4, dtype=xp.int64)，调用 differentiate 函数对其求导
        args = (xp.arange(4, dtype=xp.int64),)
        res = differentiate(f, xp.ones(4, dtype=xp.float64), rtol=1e-14,
                            order=2, args=args)

        # 定义参考的状态标志 ref_flags，并使用 xp_assert_equal 检查与 res.status 的相等性
        ref_flags = xp.asarray([eim._ECONVERGED,
                                _EERRORINCREASE,
                                eim._ECONVERR,
                                eim._EVALUEERR], dtype=xp.int32)
        xp_assert_equal(res.status, ref_flags)
    # 测试保持形状标志的功能
    def test_flags_preserve_shape(self, xp):
        # 使用 `preserve_shape` 选项进行相同的测试，以简化代码。
        rng = np.random.default_rng(5651219684984213)
        # 定义一个函数 f(x)，返回一个包含多个数组的列表
        def f(x):
            # 定义输出列表
            out = [
                x - 2.5,  # 收敛
                xp.exp(x)*rng.random(),  # 错误增加
                xp.exp(x),  # 因为 order=2 达到最大迭代次数
                xp.full_like(x, xp.nan)[()]  # 因为 NaN 停止
            ]
            return xp.stack(out)

        # 调用 differentiate 函数，传入参数并获取结果
        res = differentiate(f, xp.asarray(1, dtype=xp.float64), rtol=1e-14,
                            order=2, preserve_shape=True)

        # 定义参考的标志数组
        ref_flags = xp.asarray([eim._ECONVERGED,
                                _EERRORINCREASE,
                                eim._ECONVERR,
                                eim._EVALUEERR], dtype=xp.int32)
        # 使用 xp_assert_equal 函数断言结果的状态等于参考标志
        xp_assert_equal(res.status, ref_flags)

    # 测试 `preserve_shape` 选项
    def test_preserve_shape(self, xp):
        # 定义一个函数 f(x)，返回一个包含多个数组的列表
        def f(x):
            out = [
                x, xp.sin(3*x), x+xp.sin(10*x), xp.sin(20*x)*(x-1)**2
            ]
            return xp.stack(out)

        # 定义输入值 x
        x = xp.asarray(0.)
        # 定义参考结果 ref
        ref = xp.asarray([
            xp.asarray(1), 3*xp.cos(3*x), 1+10*xp.cos(10*x),
            20*xp.cos(20*x)*(x-1)**2 + 2*xp.sin(20*x)*(x-1)
        ])
        # 调用 differentiate 函数，传入参数并获取结果
        res = differentiate(f, x, preserve_shape=True)
        # 使用 xp_assert_close 函数断言结果的导数 df 接近于参考结果 ref
        xp_assert_close(res.df, ref)

    # 测试收敛性
    def test_convergence(self, xp):
        # 测试收敛容限的预期行为
        # 定义输入值 x
        x = xp.asarray(1., dtype=xp.float64)
        # 定义函数 f 为标准正态分布的累积分布函数
        f = special.ndtr
        # 计算参考值 ref
        ref = float(stats.norm.pdf(1.))
        # 定义关键字参数 kwargs0
        kwargs0 = dict(atol=0, rtol=0, order=4)

        # 测试绝对容限 atol 的不同设置
        kwargs = kwargs0.copy()
        kwargs['atol'] = 1e-3
        # 调用 differentiate 函数，传入参数并获取结果
        res1 = differentiate(f, x, **kwargs)
        # 使用 assert 断言结果的导数 df 与参考值 ref 的差小于 1e-3
        assert abs(res1.df - ref) < 1e-3
        kwargs['atol'] = 1e-6
        # 再次调用 differentiate 函数，传入参数并获取结果
        res2 = differentiate(f, x, **kwargs)
        # 使用 assert 断言结果的导数 df 与参考值 ref 的差小于 1e-6，且比较 res2 与 res1 的差
        assert abs(res2.df - ref) < 1e-6
        assert abs(res2.df - ref) < abs(res1.df - ref)

        # 测试相对容限 rtol 的不同设置
        kwargs = kwargs0.copy()
        kwargs['rtol'] = 1e-3
        # 调用 differentiate 函数，传入参数并获取结果
        res1 = differentiate(f, x, **kwargs)
        # 使用 assert 断言结果的导数 df 与参考值 ref 的相对差小于 1e-3
        assert abs(res1.df - ref) < 1e-3 * ref
        kwargs['rtol'] = 1e-6
        # 再次调用 differentiate 函数，传入参数并获取结果
        res2 = differentiate(f, x, **kwargs)
        # 使用 assert 断言结果的导数 df 与参考值 ref 的相对差小于 1e-6，且比较 res2 与 res1 的差
        assert abs(res2.df - ref) < 1e-6 * ref
        assert abs(res2.df - ref) < abs(res1.df - ref)
    def test_step_parameters(self, xp):
        # 测试步长因子对准确度的预期影响
        x = xp.asarray(1., dtype=xp.float64)
        # 使用 SciPy 中的正态分布的累积分布函数
        f = special.ndtr
        # 参考值为标准正态分布在 x=1 处的概率密度函数值
        ref = float(stats.norm.pdf(1.))

        # 使用不同的初始步长测试区别
        res1 = differentiate(f, x, initial_step=0.5, maxiter=1)
        res2 = differentiate(f, x, initial_step=0.05, maxiter=1)
        # 断言较小步长的结果误差应该更小
        assert abs(res2.df - ref) < abs(res1.df - ref)

        # 使用不同的步长因子测试区别
        res1 = differentiate(f, x, step_factor=2, maxiter=1)
        res2 = differentiate(f, x, step_factor=20, maxiter=1)
        # 断言较大步长因子的结果误差应该更小
        assert abs(res2.df - ref) < abs(res1.df - ref)

        # `step_factor` 可以小于1：`initial_step` 是最小步长
        kwargs = dict(order=4, maxiter=1, step_direction=0)
        res = differentiate(f, x, initial_step=0.5, step_factor=0.5, **kwargs)
        ref = differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        # 断言结果应该非常接近（相对误差小于5e-15）
        xp_assert_close(res.df, ref.df, rtol=5e-15)

        # 这是单边差分的类似测试
        kwargs = dict(order=2, maxiter=1, step_direction=1)
        res = differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        ref = differentiate(f, x, initial_step=1/np.sqrt(2), step_factor=0.5,
                                   **kwargs)
        # 断言结果应该非常接近（相对误差小于5e-15）
        xp_assert_close(res.df, ref.df, rtol=5e-15)

        # 更新步长方向为负方向的测试
        kwargs['step_direction'] = -1
        res = differentiate(f, x, initial_step=1, step_factor=2, **kwargs)
        ref = differentiate(f, x, initial_step=1/np.sqrt(2), step_factor=0.5,
                                   **kwargs)
        # 断言结果应该非常接近（相对误差小于5e-15）
        xp_assert_close(res.df, ref.df, rtol=5e-15)

    def test_step_direction(self, xp):
        # 测试 `step_direction` 的预期工作效果
        def f(x):
            y = xp.exp(x)
            y[(x < 0) + (x > 2)] = xp.nan
            return y

        x = xp.linspace(0, 2, 10)
        step_direction = xp.zeros_like(x)
        step_direction[x < 0.6], step_direction[x > 1.4] = 1, -1
        # 执行数值微分
        res = differentiate(f, x, step_direction=step_direction)
        # 断言结果与指数函数的值非常接近
        xp_assert_close(res.df, xp.exp(x))
        assert xp.all(res.success)

    def test_vectorized_step_direction_args(self, xp):
        # 测试 `step_direction` 和 `args` 的向量化是否正确
        def f(x, p):
            return x ** p

        def df(x, p):
            return p * x ** (p - 1)

        x = xp.reshape(xp.asarray([1, 2, 3, 4]), (-1, 1, 1))
        hdir = xp.reshape(xp.asarray([-1, 0, 1]), (1, -1, 1))
        p = xp.reshape(xp.asarray([2, 3]), (1, 1, -1))
        # 执行数值微分
        res = differentiate(f, x, step_direction=hdir, args=(p,))
        ref = xp.broadcast_to(df(x, p), res.df.shape)
        ref = xp.asarray(ref, dtype=xp.asarray(1.).dtype)
        # 断言结果应该非常接近
        xp_assert_close(res.df, ref)
    def test_maxiter_callback(self, xp):
        # Test behavior of `maxiter` parameter and `callback` interface

        # Initialize a scalar array with a specific floating-point value
        x = xp.asarray(0.612814, dtype=xp.float64)

        # Set the maximum number of iterations
        maxiter = 3

        # Define a function `f` that computes the standard normal cumulative distribution function
        def f(x):
            res = special.ndtr(x)
            return res

        # Default order of differentiation
        default_order = 8

        # Perform differentiation with specified parameters and get the result
        res = differentiate(f, x, maxiter=maxiter, rtol=1e-15)

        # Assert that all elements in `success` array are False
        assert not xp.any(res.success)

        # Assert that all elements in `nfev` array are as expected
        assert xp.all(res.nfev == default_order + 1 + (maxiter - 1)*2)

        # Assert that all elements in `nit` array are equal to `maxiter`
        assert xp.all(res.nit == maxiter)

        # Define a callback function for handling iterative results
        def callback(res):
            callback.iter += 1
            callback.res = res
            # Assert certain attributes of the result object
            assert hasattr(res, 'x')
            assert float(res.df) not in callback.dfs
            callback.dfs.add(float(res.df))
            assert res.status == eim._EINPROGRESS
            # Raise StopIteration after `maxiter` iterations
            if callback.iter == maxiter:
                raise StopIteration

        # Initialize properties on the callback function for tracking state
        callback.iter = -1  # callback called once before first iteration
        callback.res = None
        callback.dfs = set()

        # Perform differentiation with callback function and get the result
        res2 = differentiate(f, x, callback=callback, rtol=1e-15)

        # Compare results to ensure consistency when terminating with callback vs. maxiter
        # (except for `status`)
        for key in res.keys():
            if key == 'status':
                assert res[key] == eim._ECONVERR
                assert res2[key] == eim._ECALLBACK
            else:
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize("hdir", (-1, 0, 1))
    @pytest.mark.parametrize("x", (0.65, [0.65, 0.7]))
    @pytest.mark.parametrize("dtype", ('float16', 'float32', 'float64'))
    def test_dtype(self, hdir, x, dtype, xp):
        # Skip test for float16 if backend is not NumPy
        if dtype == 'float16' and not is_numpy(xp):
            pytest.skip('float16 not tested for alternative backends')

        # Test that dtypes are preserved when converting `x` to specified dtype
        dtype = getattr(xp, dtype)
        x = xp.asarray(x, dtype=dtype)[()]

        # Define a function `f` that computes the exponential function
        def f(x):
            assert x.dtype == dtype
            return xp.exp(x)

        # Define a callback function `callback` to assert dtypes of result attributes
        def callback(res):
            assert res.x.dtype == dtype
            assert res.df.dtype == dtype
            assert res.error.dtype == dtype

        # Perform differentiation with specified parameters and get the result
        res = differentiate(f, x, order=4, step_direction=hdir,
                                   callback=callback)

        # Assert dtypes of result attributes after differentiation
        assert res.x.dtype == dtype
        assert res.df.dtype == dtype
        assert res.error.dtype == dtype

        # Calculate epsilon for relative tolerance comparison
        eps = xp.finfo(dtype).eps
        rtol = eps**0.5 * 50 if is_torch(xp) else eps**0.5

        # Assert that `df` is close to `exp(x)` with computed relative tolerance
        xp_assert_close(res.df, xp.exp(res.x), rtol=rtol)
    # 定义一个测试方法，用于输入验证，检查不合适的输入是否会触发适当的错误消息
    def test_input_validation(self, xp):
        # 使用 xp.asarray 将整数 1 转换为特定库（如 NumPy）的数组表示形式
        one = xp.asarray(1)

        # 设置错误消息：`func` 必须是可调用的。
        message = '`func` must be callable.'
        # 使用 pytest 检查调用 differentiate 函数时传入 None 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(None, one)

        # 设置错误消息：横坐标和函数输出必须是实数。
        message = 'Abscissae and function output must be real numbers.'
        # 使用 pytest 检查调用 differentiate 函数时传入包含虚数的数组是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, xp.asarray(-4+1j))

        # 设置错误消息：当 `preserve_shape=False` 时，数组的形状必须...
        message = "When `preserve_shape=False`, the shape of the array..."
        # 使用 pytest 检查调用 differentiate 函数时传入不符合形状要求的输入是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: [1, 2, 3], xp.asarray([-2, -3]))

        # 设置错误消息：公差和步长参数必须是非负数...
        message = 'Tolerances and step parameters must be non-negative...'
        # 使用 pytest 检查调用 differentiate 函数时传入负公差值是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, atol=-1)
        # 使用 pytest 检查调用 differentiate 函数时传入非数值类型的相对公差是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, rtol='ekki')
        # 使用 pytest 检查调用 differentiate 函数时传入 None 作为初始步长是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, initial_step=None)
        # 使用 pytest 检查调用 differentiate 函数时传入非数值类型的步长因子是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, step_factor=object())

        # 设置错误消息：`maxiter` 必须是正整数。
        message = '`maxiter` must be a positive integer.'
        # 使用 pytest 检查调用 differentiate 函数时传入非整数的 `maxiter` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, maxiter=1.5)
        # 使用 pytest 检查调用 differentiate 函数时传入零作为 `maxiter` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, maxiter=0)

        # 设置错误消息：`order` 必须是正整数。
        message = '`order` must be a positive integer'
        # 使用 pytest 检查调用 differentiate 函数时传入非整数的 `order` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, order=1.5)
        # 使用 pytest 检查调用 differentiate 函数时传入零作为 `order` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, order=0)

        # 设置错误消息：`preserve_shape` 必须是 True 或 False。
        message = '`preserve_shape` must be True or False.'
        # 使用 pytest 检查调用 differentiate 函数时传入非布尔值的 `preserve_shape` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, preserve_shape='herring')

        # 设置错误消息：`callback` 必须是可调用的。
        message = '`callback` must be callable.'
        # 使用 pytest 检查调用 differentiate 函数时传入非可调用对象作为 `callback` 是否会引发 ValueError，并匹配特定错误消息
        with pytest.raises(ValueError, match=message):
            differentiate(lambda x: x, one, callback='shrubbery')
    # 定义测试特殊情况的函数，使用给定的数学库（xp）

    # 测试整数不能作为 `f` 的参数传递（否则会溢出）
    def f(x):
        # 使用数组命名空间对输入进行检查，需要 `isdtype`
        xp_test = array_namespace(x)
        # 断言输入类型是实数浮点数
        assert xp_test.isdtype(x.dtype, 'real floating')
        # 计算 x 的 99 次方减 1 的结果
        return x ** 99 - 1

    # 如果不是 torch 数学库（默认为 float32），进行测试
    if not is_torch(xp):
        # 对 f(7) 进行自动微分，预期误差较小
        res = differentiate(f, xp.asarray(7), rtol=1e-10)
        # 断言自动微分成功
        assert res.success
        # 断言微分结果接近预期值
        xp_assert_close(res.df, xp.asarray(99 * 7. ** 98))

    # 测试多项式函数在正确迭代次数下是否收敛
    for n in range(6):
        # 定义测试函数 f(x) = 2 * x^n
        x = xp.asarray(1.5, dtype=xp.float64)
        def f(x):
            return 2 * x**n

        # 计算 f(x) 的导数参考值
        ref = 2 * n * x**(n - 1)

        # 使用最大迭代次数为 1 进行自动微分，测试低阶多项式
        res = differentiate(f, x, maxiter=1, order=max(1, n))
        # 断言自动微分结果接近参考值
        xp_assert_close(res.df, ref, rtol=1e-15)
        # 断言错误估计为 NaN
        xp_assert_equal(res.error, xp.asarray(xp.nan, dtype=xp.float64))

        # 使用默认迭代次数进行自动微分，测试高阶多项式
        res = differentiate(f, x, order=max(1, n))
        # 断言自动微分成功
        assert res.success
        # 断言迭代次数为 2
        assert res.nit == 2
        # 断言自动微分结果接近参考值
        xp_assert_close(res.df, ref, rtol=1e-15)

    # 测试标量 `args`（不在元组中）
    def f(x, c):
        return c * x - 1

    # 对 f(2, 3) 进行自动微分
    res = differentiate(f, xp.asarray(2), args=xp.asarray(3))
    # 断言自动微分结果接近预期值
    xp_assert_close(res.df, xp.asarray(3.))

# 不需要在多个后端上运行测试，因为预期是失败的
@pytest.mark.skip_xp_backends(np_only=True)
@pytest.mark.xfail
@pytest.mark.parametrize("case", (  # 函数, 评估点
    (lambda x: (x - 1) ** 3, 1),
    (lambda x: np.where(x > 1, (x - 1) ** 5, (x - 1) ** 3), 1)
))
# 测试特定情况下的鞍点问题
def test_saddle_gh18811(self, case):
    # 使用默认设置进行自动微分，在真实导数为零的情况下可能不会收敛
    # 通过指定一个小的 `atol` 可以缓解这个问题，参见 gh-18811 中的讨论
    atol = 1e-16
    # 进行自动微分，使用给定的步进方向和 `atol`
    res = differentiate(*case, step_direction=[-1, 0, 1], atol=atol)
    # 断言所有结果都成功
    assert np.all(res.success)
    # 断言所有结果的导数接近零，使用指定的 `atol`
    assert_allclose(res.df, 0, atol=atol)
`
class TestJacobian:

    # Example functions and Jacobians from Wikipedia:
    # https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Examples

    # 定义函数 f1，接受参数 z，并返回包含两个元素的列表
    def f1(z):
        x, y = z
        return [x ** 2 * y, 5 * x + np.sin(y)]

    # 定义函数 df1，接受参数 z，并返回一个包含两个列表的列表
    def df1(z):
        x, y = z
        return [[2 * x * y, x ** 2], [np.full_like(x, 5), np.cos(y)]]

    # 设定属性 mn 和 ref 分别为元组 (2, 2) 和函数 df1
    f1.mn = 2, 2  # type: ignore[attr-defined]
    f1.ref = df1  # type: ignore[attr-defined]

    # 定义函数 f2，接受参数 z，并返回一个包含两个元素的列表
    def f2(z):
        r, phi = z
        return [r * np.cos(phi), r * np.sin(phi)]

    # 定义函数 df2，接受参数 z，并返回一个包含两个列表的列表
    def df2(z):
        r, phi = z
        return [[np.cos(phi), -r * np.sin(phi)],
                [np.sin(phi), r * np.cos(phi)]]

    # 设定属性 mn 和 ref 分别为元组 (2, 2) 和函数 df2
    f2.mn = 2, 2  # type: ignore[attr-defined]
    f2.ref = df2  # type: ignore[attr-defined]

    # 定义函数 f3，接受参数 z，并返回一个包含三个元素的列表
    def f3(z):
        r, phi, th = z
        return [r * np.sin(phi) * np.cos(th), r * np.sin(phi) * np.sin(th),
                r * np.cos(phi)]

    # 定义函数 df3，接受参数 z，并返回一个包含三个列表的列表
    def df3(z):
        r, phi, th = z
        return [[np.sin(phi) * np.cos(th), r * np.cos(phi) * np.cos(th),
                 -r * np.sin(phi) * np.sin(th)],
                [np.sin(phi) * np.sin(th), r * np.cos(phi) * np.sin(th),
                 r * np.sin(phi) * np.cos(th)],
                [np.cos(phi), -r * np.sin(phi), np.zeros_like(r)]]

    # 设定属性 mn 和 ref 分别为元组 (3, 3) 和函数 df3
    f3.mn = 3, 3  # type: ignore[attr-defined]
    f3.ref = df3  # type: ignore[attr-defined]

    # 定义函数 f4，接受参数 x，并返回一个包含四个元素的列表
    def f4(x):
        x1, x2, x3 = x
        return [x1, 5 * x3, 4 * x2 ** 2 - 2 * x3, x3 * np.sin(x1)]

    # 定义函数 df4，接受参数 x，并返回一个包含四个列表的列表
    def df4(x):
        x1, x2, x3 = x
        one = np.ones_like(x1)
        return [[one, 0 * one, 0 * one],
                [0 * one, 0 * one, 5 * one],
                [0 * one, 8 * x2, -2 * one],
                [x3 * np.cos(x1), 0 * one, np.sin(x1)]]

    # 设定属性 mn 和 ref 分别为元组 (3, 4) 和函数 df4
    f4.mn = 3, 4  # type: ignore[attr-defined]
    f4.ref = df4  # type: ignore[attr-defined]

    # 定义函数 f5，接受参数 x，并返回一个包含三个元素的列表
    def f5(x):
        x1, x2, x3 = x
        return [5 * x2, 4 * x1 ** 2 - 2 * np.sin(x2 * x3), x2 * x3]

    # 定义函数 df5，接受参数 x，并返回一个包含三个列表的列表
    def df5(x):
        x1, x2, x3 = x
        one = np.ones_like(x1)
        return [[0 * one, 5 * one, 0 * one],
                [8 * x1, -2 * x3 * np.cos(x2 * x3), -2 * x2 * np.cos(x2 * x3)],
                [0 * one, x3, x2]]

    # 设定属性 mn 和 ref 分别为元组 (3, 3) 和函数 df5
    f5.mn = 3, 3  # type: ignore[attr-defined]
    f5.ref = df5  # type: ignore[attr-defined]

    # 设定 rosen 为 optimize.rosen，其属性 mn 为元组 (5, 1)，ref 为 optimize.rosen_der
    rosen = optimize.rosen
    rosen.mn = 5, 1  # type: ignore[attr-defined]
    rosen.ref = optimize.rosen_der  # type: ignore[attr-defined]

    # 使用 pytest 提供的参数化装饰器，对 size 和 func 进行参数化测试
    @pytest.mark.parametrize('size', [(), (6,), (2, 3)])
    @pytest.mark.parametrize('func', [f1, f2, f3, f4, f5, rosen])
    # 定义测试方法 test_examples，接受参数 self, size, func
    def test_examples(self, size, func):
        # 使用指定种子创建随机数生成器 rng
        rng = np.random.default_rng(458912319542)
        # 获取 func 的 mn 属性的值赋给 m, n
        m, n = func.mn
        # 使用 rng 创建形状为 (m,) + size 的随机数组 x
        x = rng.random(size=(m,) + size)
        # 计算函数 func 在 x 处的雅可比矩阵的 df 属性
        res = jacobian(func, x).df
        # 获取函数 func 的参考值 ref
        ref = func.ref(x)
        # 使用 np.testing.assert_allclose 进行数值接近性检查
        np.testing.assert_allclose(res, ref, atol=1e-10)
    # 定义测试方法，用于测试输入验证
    def test_iv(self):
        # 检查输入验证
        message = "Argument `x` must be at least 1-D."
        # 使用 pytest 来检测是否会引发 ValueError 异常，并匹配指定错误信息
        with pytest.raises(ValueError, match=message):
            # 调用 jacobian 函数，并传入参数 np.sin, 1, atol=-1，期望引发异常
            jacobian(np.sin, 1, atol=-1)

        # 确认其他参数是否传递给 `_derivative` 函数，该函数会引发适当的错误信息
        x = np.ones(3)
        func = optimize.rosen
        message = 'Tolerances and step parameters must be non-negative scalars.'
        # 检查不合法参数情况下是否会引发 ValueError 异常，并匹配指定错误信息
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, atol=-1)
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, rtol=-1)
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, initial_step=-1)
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, step_factor=-1)

        message = '`order` must be a positive integer.'
        # 检查 order 参数是否为正整数，若不是则引发 ValueError 异常，并匹配指定错误信息
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, order=-1)

        message = '`maxiter` must be a positive integer.'
        # 检查 maxiter 参数是否为正整数，若不是则引发 ValueError 异常，并匹配指定错误信息
        with pytest.raises(ValueError, match=message):
            jacobian(func, x, maxiter=-1)
```