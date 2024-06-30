# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_chandrupatla.py`

```
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less

from scipy import stats, special
import scipy._lib._elementwise_iterative_method as eim
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (array_namespace, xp_assert_close, xp_assert_equal,
                                   xp_assert_less, xp_minimum, is_numpy, is_cupy)

from scipy.optimize._chandrupatla import (_chandrupatla_minimize,
                                          _chandrupatla as _chandrupatla_root)
from scipy.optimize._tstutils import _CHANDRUPATLA_TESTS

from itertools import permutations
from .test_zeros import TestScalarRootFinders

# 定义多项式函数 f1 到 f5
def f1(x):
    return 100*(1 - x**3.)**2 + (1-x**2.) + 2*(1-x)**2.


def f2(x):
    return 5 + (x - 2.)**6


def f3(x):
    return np.exp(x) - 5*x


def f4(x):
    return x**5. - 5*x**3. - 20.*x + 5.


def f5(x):
    return 8*x**3 - 2*x**2 - 7*x + 3

# 定义函数 _bracket_minimum，用于寻找函数的局部最小值区间
def _bracket_minimum(func, x1, x2):
    phi = 1.61803398875  # 黄金分割比例
    maxiter = 100  # 最大迭代次数
    f1 = func(x1)  # 计算函数在 x1 处的值
    f2 = func(x2)  # 计算函数在 x2 处的值
    step = x2 - x1  # 初始步长为 x2 - x1

    # 交换 x1 和 x2，确保 f1 对应较小的函数值
    x1, x2, f1, f2, step = ((x2, x1, f2, f1, -step) if f2 > f1
                            else (x1, x2, f1, f2, step))

    # 开始迭代寻找最小值区间
    for i in range(maxiter):
        step *= phi  # 增加步长
        x3 = x2 + step  # 计算新的候选点 x3
        f3 = func(x3)  # 计算函数在 x3 处的值
        if f3 < f2:
            x1, x2, f1, f2 = x2, x3, f2, f3  # 更新最小值区间和函数值
        else:
            break

    # 返回找到的最小值区间的端点和相应的函数值
    return x1, x2, x3, f1, f2, f3

# 定义测试用例 cases，包含函数和其预期的最小值
cases = [
    (f1, -1, 11),
    (f1, -2, 13),
    (f1, -4, 13),
    (f1, -8, 15),
    (f1, -16, 16),
    (f1, -32, 19),
    (f1, -64, 20),
    (f1, -128, 21),
    (f1, -256, 21),
    (f1, -512, 19),
    (f1, -1024, 24),
    (f2, -1, 8),
    (f2, -2, 6),
    (f2, -4, 6),
    (f2, -8, 7),
    (f2, -16, 8),
    (f2, -32, 8),
    (f2, -64, 9),
    (f2, -128, 11),
    (f2, -256, 13),
    (f2, -512, 12),
    (f2, -1024, 13),
    (f3, -1, 11),
    (f3, -2, 11),
    (f3, -4, 11),
    (f3, -8, 10),
    (f3, -16, 14),
    (f3, -32, 12),
    (f3, -64, 15),
    (f3, -128, 18),
    (f3, -256, 18),
    (f3, -512, 19),
    (f3, -1024, 19),
    (f4, -0.05, 9),
    (f4, -0.10, 11),
    (f4, -0.15, 11),
    (f4, -0.20, 11),
    (f4, -0.25, 11),
    (f4, -0.30, 9),
    (f4, -0.35, 9),
    (f4, -0.40, 9),
    (f4, -0.45, 10),
    (f4, -0.50, 10),
    (f4, -0.55, 10),
    (f5, -0.05, 6),
    (f5, -0.10, 7),
    (f5, -0.15, 8),
    (f5, -0.20, 10),
    (f5, -0.25, 9),
    (f5, -0.30, 8),
    (f5, -0.35, 7),
    (f5, -0.40, 7),
    (f5, -0.45, 9),
    (f5, -0.50, 9),
    (f5, -0.55, 8)
]

# 定义测试类 TestChandrupatlaMinimize，用于测试 _chandrupatla_minimize 函数
class TestChandrupatlaMinimize:

    # 定义测试函数 f，用于测试 _chandrupatla_minimize 函数的局部最小化能力
    def f(self, x, loc):
        dist = stats.norm()
        return -dist.pdf(x - loc)

    # 使用 pytest 的参数化装饰器指定不同的 loc 参数进行多次测试
    @pytest.mark.parametrize('loc', [0.6, np.linspace(-1.05, 1.05, 10)])
    # 定义一个测试方法，用于基本测试
    def test_basic(self, loc):
        # 使用 _chandrupatla_minimize 函数找到正态分布的众数。
        # 比较众数与位置参数 loc 的值，并比较众数处的概率密度函数值与预期的概率密度函数值。
        res = _chandrupatla_minimize(self.f, -5, 0, 5, args=(loc,))
        
        # 将参考值设为 loc
        ref = loc
        
        # 断言找到的最小值 res.x 与参考值 ref 之间的接近程度
        np.testing.assert_allclose(res.x, ref, rtol=1e-6)
        
        # 断言找到的最小函数值 res.fun 与标准正态分布在零点的概率密度函数值接近
        np.testing.assert_allclose(res.fun, -stats.norm.pdf(0), atol=0, rtol=0)
        
        # 断言 res.x 的形状与 ref 的形状相同
        assert res.x.shape == np.shape(ref)

    # 使用 pytest.mark.parametrize 装饰器为 shape 参数化测试用例，shape 可以是空元组、长度为 12 的元组、3x4 的矩阵、3x2x2 的三维矩阵
    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    # 定义一个测试向量化功能的方法，用于不同形状的输入测试正确功能、输出形状和数据类型。
    def test_vectorization(self, shape):
        # 如果 shape 存在，则创建一个均匀分布的 loc 数组，否则设置 loc 为 0.6
        loc = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        # 将 loc 组成元组 args
        args = (loc,)

        # 定义一个使用 np.vectorize 装饰器的内部函数，用于将单个 loc_single 传递给 _chandrupatla_minimize 函数
        @np.vectorize
        def chandrupatla_single(loc_single):
            return _chandrupatla_minimize(self.f, -5, 0, 5, args=(loc_single,))

        # 定义一个函数 f，用于增加函数评估的计数，并调用 self.f 方法
        def f(*args, **kwargs):
            f.f_evals += 1
            return self.f(*args, **kwargs)
        # 初始化函数评估计数为 0
        f.f_evals = 0

        # 使用 _chandrupatla_minimize 函数对 f 进行最小化，传入参数 args
        res = _chandrupatla_minimize(f, -5, 0, 5, args=args)
        
        # 调用 chandrupatla_single 函数并展开结果，得到参考值 refs
        refs = chandrupatla_single(loc).ravel()

        # 提取参考值中的 x 属性到列表 ref_x 中
        ref_x = [ref.x for ref in refs]
        # 断言 res 的 x 属性与 ref_x 接近
        assert_allclose(res.x.ravel(), ref_x)
        # 断言 res 的 x 属性的形状与 shape 相同
        assert_equal(res.x.shape, shape)

        # 提取参考值中的 fun 属性到列表 ref_fun 中
        ref_fun = [ref.fun for ref in refs]
        # 断言 res 的 fun 属性与 ref_fun 接近
        assert_allclose(res.fun.ravel(), ref_fun)
        # 断言 res 的 fun 属性的形状与 shape 相同
        assert_equal(res.fun.shape, shape)
        # 断言 res 的 fun 属性与 self.f 函数计算的结果相同
        assert_equal(res.fun, self.f(res.x, *args))

        # 提取参考值中的 success 属性到列表 ref_success 中
        ref_success = [ref.success for ref in refs]
        # 断言 res 的 success 属性与 ref_success 相同
        assert_equal(res.success.ravel(), ref_success)
        # 断言 res 的 success 属性的形状与 shape 相同
        assert_equal(res.success.shape, shape)
        # 断言 res 的 success 属性的数据类型是布尔类型
        assert np.issubdtype(res.success.dtype, np.bool_)

        # 提取参考值中的 status 属性到列表 ref_flag 中
        ref_flag = [ref.status for ref in refs]
        # 断言 res 的 status 属性与 ref_flag 相同
        assert_equal(res.status.ravel(), ref_flag)
        # 断言 res 的 status 属性的形状与 shape 相同
        assert_equal(res.status.shape, shape)
        # 断言 res 的 status 属性的数据类型是整数类型
        assert np.issubdtype(res.status.dtype, np.integer)

        # 提取参考值中的 nfev 属性到列表 ref_nfev 中
        ref_nfev = [ref.nfev for ref in refs]
        # 断言 res 的 nfev 属性与 ref_nfev 相同
        assert_equal(res.nfev.ravel(), ref_nfev)
        # 断言 res 的 nfev 属性的最大值与 f 的评估次数相同
        assert_equal(np.max(res.nfev), f.f_evals)
        # 断言 res 的 nfev 属性的形状与 res 的 fun 属性相同
        assert_equal(res.nfev.shape, res.fun.shape)
        # 断言 res 的 nfev 属性的数据类型是整数类型
        assert np.issubdtype(res.nfev.dtype, np.integer)

        # 提取参考值中的 nit 属性到列表 ref_nit 中
        ref_nit = [ref.nit for ref in refs]
        # 断言 res 的 nit 属性与 ref_nit 相同
        assert_equal(res.nit.ravel(), ref_nit)
        # 断言 res 的 nit 属性的最大值与 f 的评估次数减去 3 相同
        assert_equal(np.max(res.nit), f.f_evals-3)
        # 断言 res 的 nit 属性的形状与 res 的 fun 属性相同
        assert_equal(res.nit.shape, res.fun.shape)
        # 断言 res 的 nit 属性的数据类型是整数类型
        assert np.issubdtype(res.nit.dtype, np.integer)

        # 提取参考值中的 xl 属性到列表 ref_xl 中
        ref_xl = [ref.xl for ref in refs]
        # 断言 res 的 xl 属性与 ref_xl 接近
        assert_allclose(res.xl.ravel(), ref_xl)
        # 断言 res 的 xl 属性的形状与 shape 相同
        assert_equal(res.xl.shape, shape)

        # 提取参考值中的 xm 属性到列表 ref_xm 中
        ref_xm = [ref.xm for ref in refs]
        # 断言 res 的 xm 属性与 ref_xm 接近
        assert_allclose(res.xm.ravel(), ref_xm)
        # 断言 res 的 xm 属性的形状与 shape 相同
        assert_equal(res.xm.shape, shape)

        # 提取参考值中的 xr 属性到列表 ref_xr 中
        ref_xr = [ref.xr for ref in refs]
        # 断言 res 的 xr 属性与 ref_xr 接近
        assert_allclose(res.xr.ravel(), ref_xr)
        # 断言 res 的 xr 属性的形状与 shape 相同
        assert_equal(res.xr.shape, shape)

        # 提取参考值中的 fl 属性到列表 ref_fl 中
        ref_fl = [ref.fl for ref in refs]
        # 断言 res 的 fl 属性与 ref_fl 接近
        assert_allclose(res.fl.ravel(), ref_fl)
        # 断言 res 的 fl 属性的形状与 shape 相同
        assert_equal(res.fl.shape, shape)
        # 断言 res 的 fl 属性与 self.f 函数计算的结果接近
        assert_allclose(res.fl, self.f(res.xl, *args))

        # 提取参考值中的 fm 属性到列表 ref_fm 中
        ref_fm = [ref.fm for ref in refs]
        # 断言 res 的 fm 属性与 ref_fm 接近
        assert_allclose(res.fm.ravel(), ref_fm)
        # 断言 res 的 fm 属性的形状与 shape 相同
        assert_equal(res.fm.shape, shape)
        # 断言 res 的 fm 属性与 self.f 函数计算的结果接近
        assert_allclose(res.fm, self.f(res.xm, *args))

        # 提取参考值中的 fr 属性到列表 ref_fr 中
        ref_fr = [ref.fr for ref in refs]
        # 断言 res 的 fr 属性与 ref_fr 接近
        assert_allclose(res.fr.ravel(), ref_fr)
        # 断言 res 的 fr 属性的形状与 shape 相同
        assert_equal(res.fr.shape, shape)
        # 断言 res 的 fr 属性与 self.f 函数计算的结果接近
        assert_allclose(res.fr, self.f(res.xr, *args))
    def test_flags(self):
        # 测试不同状态标志的案例；展示可以同时产生所有状态标志。
        def f(xs, js):
            # 定义一组函数列表，每个函数都接收一个参数 x，并返回计算结果
            funcs = [lambda x: (x - 2.5) ** 2,
                     lambda x: x - 10,
                     lambda x: (x - 2.5) ** 4,
                     lambda x: np.nan]

            # 对于输入的 xs 和 js，使用 funcs 中的函数来计算结果
            return [funcs[j](x) for x, j in zip(xs, js)]

        args = (np.arange(4, dtype=np.int64),)

        # 调用 _chandrupatla_minimize 函数进行优化计算
        res = _chandrupatla_minimize(f, [0]*4, [2]*4, [np.pi]*4, args=args,
                                     maxiter=10)

        # 定义参考的状态标志数组
        ref_flags = np.array([eim._ECONVERGED,
                              eim._ESIGNERR,
                              eim._ECONVERR,
                              eim._EVALUEERR])
        # 断言优化结果的状态与参考标志数组一致
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        # 测试收敛容差是否按预期工作
        rng = np.random.default_rng(2585255913088665241)
        p = rng.random(size=3)
        bracket = (-5, 0, 5)
        args = (p,)
        kwargs0 = dict(args=args, xatol=0, xrtol=0, fatol=0, frtol=0)

        kwargs = kwargs0.copy()
        # 设置 xatol 容差为 1e-3 进行优化计算
        kwargs['xatol'] = 1e-3
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j1 = abs(res1.xr - res1.xl)
        # 断言结果的区间长度 j1 小于 4 倍 xatol
        assert_array_less(j1, 4*kwargs['xatol'])
        # 重新设置 xatol 容差为 1e-6 进行优化计算
        kwargs['xatol'] = 1e-6
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j2 = abs(res2.xr - res2.xl)
        # 断言结果的区间长度 j2 小于 4 倍 xatol，并且比 j1 小
        assert_array_less(j2, 4*kwargs['xatol'])
        assert_array_less(j2, j1)

        kwargs = kwargs0.copy()
        # 设置 xrtol 相对容差为 1e-3 进行优化计算
        kwargs['xrtol'] = 1e-3
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j1 = abs(res1.xr - res1.xl)
        # 断言结果的区间长度 j1 小于 4 倍 xrtol 乘以结果 x 的绝对值
        assert_array_less(j1, 4*kwargs['xrtol']*abs(res1.x))
        # 重新设置 xrtol 相对容差为 1e-6 进行优化计算
        kwargs['xrtol'] = 1e-6
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j2 = abs(res2.xr - res2.xl)
        # 断言结果的区间长度 j2 小于 4 倍 xrtol 乘以结果 x 的绝对值，并且比 j1 小
        assert_array_less(j2, 4*kwargs['xrtol']*abs(res2.x))
        assert_array_less(j2, j1)

        kwargs = kwargs0.copy()
        # 设置 fatol 绝对容差为 1e-3 进行优化计算
        kwargs['fatol'] = 1e-3
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h1 = abs(res1.fl - 2 * res1.fm + res1.fr)
        # 断言结果的 h1 小于 2 倍 fatol
        assert_array_less(h1, 2*kwargs['fatol'])
        # 重新设置 fatol 绝对容差为 1e-6 进行优化计算
        kwargs['fatol'] = 1e-6
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h2 = abs(res2.fl - 2 * res2.fm + res2.fr)
        # 断言结果的 h2 小于 2 倍 fatol，并且比 h1 小
        assert_array_less(h2, 2*kwargs['fatol'])
        assert_array_less(h2, h1)

        kwargs = kwargs0.copy()
        # 设置 frtol 相对容差为 1e-3 进行优化计算
        kwargs['frtol'] = 1e-3
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h1 = abs(res1.fl - 2 * res1.fm + res1.fr)
        # 断言结果的 h1 小于 2 倍 frtol 乘以结果函数值的绝对值
        assert_array_less(h1, 2*kwargs['frtol']*abs(res1.fun))
        # 重新设置 frtol 相对容差为 1e-6 进行优化计算
        kwargs['frtol'] = 1e-6
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h2 = abs(res2.fl - 2 * res2.fm + res2.fr)
        # 断言结果的 h2 小于 2 倍 frtol 乘以结果函数值的绝对值，并且比 h1 小
        assert_array_less(h2, 2*kwargs['frtol']*abs(res2.fun))
        assert_array_less(h2, h1)
    def test_maxiter_callback(self):
        # Test behavior of `maxiter` parameter and `callback` interface

        # 设置本地变量 loc 为 0.612814
        loc = 0.612814
        # 定义一个包含三个元素的元组 bracket，值为 (-5, 0, 5)
        bracket = (-5, 0, 5)
        # 设置最大迭代次数 maxiter 为 5
        maxiter = 5

        # 调用 _chandrupatla_minimize 函数，传入参数和回调函数
        res = _chandrupatla_minimize(self.f, *bracket, args=(loc,),
                                     maxiter=maxiter)
        # 断言结果对象的 success 属性都为 False
        assert not np.any(res.success)
        # 断言结果对象的 nfev 属性所有元素等于 maxiter+3
        assert np.all(res.nfev == maxiter+3)
        # 断言结果对象的 nit 属性所有元素等于 maxiter
        assert np.all(res.nit == maxiter)

        # 定义回调函数 callback
        def callback(res):
            # 更新 callback.iter 属性，表示迭代次数加一
            callback.iter += 1
            # 更新 callback.res 属性为当前结果对象 res
            callback.res = res
            # 断言结果对象 res 包含属性 'x'
            assert hasattr(res, 'x')
            if callback.iter == 0:
                # 当迭代次数为 0 时，断言回调函数首次调用时的 bracket 初始状态
                assert (res.xl, res.xm, res.xr) == bracket
            else:
                # 当迭代次数不为 0 时，检查 res.xl 或 res.xr 是否有变化
                changed_xr = (res.xl == callback.xl) & (res.xr != callback.xr)
                changed_xl = (res.xl != callback.xl) & (res.xr == callback.xr)
                assert np.all(changed_xr | changed_xl)

            # 更新 callback.xl 和 callback.xr 为当前结果对象的 xl 和 xr 属性
            callback.xl = res.xl
            callback.xr = res.xr
            # 断言结果对象的 status 属性等于 eim._EINPROGRESS
            assert res.status == eim._EINPROGRESS
            # 断言计算 self.f(res.xl, loc) 的结果与 res.fl 属性相等
            assert_equal(self.f(res.xl, loc), res.fl)
            # 断言计算 self.f(res.xm, loc) 的结果与 res.fm 属性相等
            assert_equal(self.f(res.xm, loc), res.fm)
            # 断言计算 self.f(res.xr, loc) 的结果与 res.fr 属性相等
            assert_equal(self.f(res.xr, loc), res.fr)
            # 断言计算 self.f(res.x, loc) 的结果与 res.fun 属性相等
            assert_equal(self.f(res.x, loc), res.fun)
            if callback.iter == maxiter:
                # 当迭代次数达到 maxiter 时，抛出 StopIteration 异常
                raise StopIteration

        # 初始化 callback 函数的 xl 和 xr 属性为 NaN
        callback.xl = np.nan
        callback.xr = np.nan
        # 设置 callback.iter 属性为 -1，表示在第一次迭代之前调用 callback 一次
        callback.iter = -1  # callback called once before first iteration
        callback.res = None

        # 再次调用 _chandrupatla_minimize 函数，传入参数和定义好的 callback 函数
        res2 = _chandrupatla_minimize(self.f, *bracket, args=(loc,),
                                      callback=callback)

        # 断言使用 callback 终止的结果与使用 maxiter 终止的结果相同（除了 status 属性）
        for key in res.keys():
            if key == 'status':
                # 断言第一个结果对象的 status 属性为 eim._ECONVERR
                assert res[key] == eim._ECONVERR
                # 断言 callback 函数返回的结果对象的 status 属性为 eim._EINPROGRESS
                assert callback.res[key] == eim._EINPROGRESS
                # 断言第二个结果对象的 status 属性为 eim._ECALLBACK
                assert res2[key] == eim._ECALLBACK
            else:
                # 断言第二个结果对象和 callback 返回的结果对象的其它属性与第一个结果对象相等
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize('case', cases)
    def test_nit_expected(self, case):
        # Test that `_chandrupatla` implements Chandrupatla's algorithm:
        # in all 55 test cases, the number of iterations performed
        # matches the number reported in the original paper.
        func, x1, nit = case

        # Find bracket using the algorithm in the paper
        # 使用论文中的算法找到 bracket
        step = 0.2
        x2 = x1 + step
        x1, x2, x3, f1, f2, f3 = _bracket_minimum(func, x1, x2)

        # Use tolerances from original paper
        # 使用论文中定义的容差值
        xatol = 0.0001
        fatol = 0.000001
        xrtol = 1e-16
        frtol = 1e-16

        # 调用 _chandrupatla_minimize 函数，传入参数和定义的容差值
        res = _chandrupatla_minimize(func, x1, x2, x3, xatol=xatol,
                                     fatol=fatol, xrtol=xrtol, frtol=frtol)
        # 断言结果对象的 nit 属性等于预期的迭代次数 nit
        assert_equal(res.nit, nit)

    @pytest.mark.parametrize("loc", (0.65, [0.65, 0.7]))
    @pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
    def test_dtype(self, loc, dtype):
        # Test that dtypes are preserved
        loc = dtype(loc)  # 将输入的 loc 转换为指定的 dtype 类型

        def f(x, loc):
            assert x.dtype == dtype  # 断言 x 的数据类型与 dtype 相同
            return ((x - loc) ** 2).astype(dtype)  # 计算并返回 (x - loc)^2，并转换为 dtype 类型

        # 使用 _chandrupatla_minimize 函数进行最小化优化
        res = _chandrupatla_minimize(f, dtype(-3), dtype(1), dtype(5),
                                     args=(loc,))
        assert res.x.dtype == dtype  # 断言优化结果 res.x 的数据类型为 dtype
        assert_allclose(res.x, loc, rtol=np.sqrt(np.finfo(dtype).eps))  # 断言优化结果 res.x 与 loc 的接近程度

    def test_input_validation(self):
        # Test input validation for appropriate error messages

        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(None, -4, 0, 4)  # 断言调用时输入 None 会引发 ValueError

        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4+1j, 0, 4)  # 断言调用时输入复数会引发 ValueError

        message = "shape mismatch: objects cannot be broadcast"
        # raised by `np.broadcast, but the traceback is readable IMO
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, [-2, -3], [0, 0], [3, 4, 5])  # 断言调用时形状不匹配的输入会引发 ValueError

        message = "The shape of the array returned by `func` must be the same"
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: [x[0], x[1], x[1]], [-3, -3],
                                   [0, 0], [5, 5])  # 断言调用时返回数组形状不匹配会引发 ValueError

        message = 'Tolerances must be non-negative scalars.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, xatol=-1)  # 断言调用时负的容差会引发 ValueError
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, xrtol=np.nan)  # 断言调用时非数字的容差会引发 ValueError
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, fatol='ekki')  # 断言调用时非数字的容差会引发 ValueError
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, frtol=np.nan)  # 断言调用时非数字的容差会引发 ValueError

        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, maxiter=1.5)  # 断言调用时非整数的最大迭代次数会引发 ValueError
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, maxiter=-1)  # 断言调用时负的最大迭代次数会引发 ValueError

        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, callback='shrubbery')  # 断言调用时非可调用对象的回调会引发 ValueError

    def test_bracket_order(self):
        # Confirm that order of points in bracket doesn't matter
        loc = np.linspace(-1, 1, 6)[:, np.newaxis]  # 创建一个列向量 loc
        brackets = np.array(list(permutations([-5, 0, 5]))).T  # 生成排列的 brackets 组合
        res = _chandrupatla_minimize(self.f, *brackets, args=(loc,))  # 使用 _chandrupatla_minimize 函数进行最小化优化
        # 断言 res.x 接近 loc 或者 res.fun 等于 self.f(loc, loc)
        assert np.all(np.isclose(res.x, loc) | (res.fun == self.f(loc, loc)))
        ref = res.x[:, 0]  # 参考值应该是 res.x 的第一列
        assert_allclose(*np.broadcast_arrays(res.x.T, ref), rtol=1e-15)  # 断言 res.x 的广播形状与参考值的接近程度
    def test_special_cases(self):
        # 测试边界情况和特殊情况

        # 定义函数 `f`，检查参数 `x` 是否为浮点数类型，然后计算 `(x-1) ** 100`
        # 如果参数 `x` 是整数类型会导致溢出
        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return (x-1) ** 100

        # 忽略无效错误状态，并调用 `_chandrupatla_minimize` 函数进行优化
        with np.errstate(invalid='ignore'):
            res = _chandrupatla_minimize(f, -7, 0, 8, fatol=0, frtol=0)
        # 断言优化成功
        assert res.success
        # 断言优化后的结果 `res.x` 与 1 的相对误差小于 1e-3
        assert_allclose(res.x, 1, rtol=1e-3)
        # 断言优化后的函数值 `res.fun` 等于 0
        assert_equal(res.fun, 0)

        # 定义函数 `f`，计算 `(x-1)**2`
        def f(x):
            return (x-1)**2

        # 调用 `_chandrupatla_minimize` 函数进行优化，传入相同的参数 `1, 1, 1`
        res = _chandrupatla_minimize(f, 1, 1, 1)
        # 断言优化成功
        assert res.success
        # 断言优化后的最优解 `res.x` 等于 1
        assert_equal(res.x, 1)

        # 定义函数 `f`，计算 `(x-1)**2`
        def f(x):
            return (x-1)**2

        # 定义初始的区间 `bracket = (-3, 1.1, 5)`
        bracket = (-3, 1.1, 5)
        # 调用 `_chandrupatla_minimize` 函数进行优化，设置 `maxiter=0`
        res = _chandrupatla_minimize(f, *bracket, maxiter=0)
        # 断言返回的最优解区间 `res.xl, res.xr` 等于初始的 `bracket`
        assert res.xl, res.xr == bracket
        # 断言迭代次数 `res.nit` 等于 0
        assert res.nit == 0
        # 断言函数评估次数 `res.nfev` 等于 3
        assert res.nfev == 3
        # 断言状态 `res.status` 等于 -2
        assert res.status == -2
        # 断言最优解 `res.x` 等于 1.1，目前为止的最佳结果
        assert res.x == 1.1

        # 定义函数 `f`，带参数 `c`，计算 `(x-c)**2 - 1`
        def f(x, c):
            return (x-c)**2 - 1

        # 调用 `_chandrupatla_minimize` 函数进行优化，传入参数 `args=1/3`
        res = _chandrupatla_minimize(f, -1, 0, 1, args=1/3)
        # 断言优化后的最优解 `res.x` 与 1/3 的相对误差小于默认容差
        assert_allclose(res.x, 1/3)

        # 定义函数 `f`，计算 `-np.sin(x)`
        def f(x):
            return -np.sin(x)

        # 调用 `_chandrupatla_minimize` 函数进行优化，设置所有容差参数为 0
        res = _chandrupatla_minimize(f, 0, 1, np.pi, xatol=0, xrtol=0,
                                     fatol=0, frtol=0)
        # 断言优化成功
        assert res.success
        # 断言找到的最优解 `res.xl, res.xm, res.xr` 符合预期的最小值
        assert res.xl < res.xm < res.xr
        # 断言在浮点运算精度内，函数值 `f(res.xl), f(res.xm), f(res.xr)` 相等
        assert f(res.xl) == f(res.xm) == f(res.xr)
# 使用装饰器使该类与数组API兼容
@array_api_compatible
# 使用pytest装饰器标记，跳过与XP后端相关的测试
@pytest.mark.usefixtures("skip_xp_backends")
# 使用pytest装饰器标记，跳过与特定XP后端相关的测试，提供跳过的理由
@pytest.mark.skip_xp_backends('array_api_strict', 'jax.numpy',
                              reasons=['Currently uses fancy indexing assignment.',
                                       'JAX arrays do not support item assignment.'])
# 定义一个测试类，继承自TestScalarRootFinders类
class TestChandrupatla(TestScalarRootFinders):

    # 定义一个函数f，接受q和p作为参数，返回special.ndtr(q) - p的结果
    def f(self, q, p):
        return special.ndtr(q) - p

    # 使用pytest装饰器标记，参数化测试用例p，分别使用0.6和np.linspace(-0.05, 1.05, 10)作为参数
    @pytest.mark.parametrize('p', [0.6, np.linspace(-0.05, 1.05, 10)])
    # 定义一个测试方法test_basic，接受参数p和xp
    def test_basic(self, p, xp):
        # 变量a和b分别被转换为xp数组，并初始化为-5.和5.
        a, b = xp.asarray(-5.), xp.asarray(5.)
        # 调用_chandrupatla_root函数，传入self.f作为函数参数，a和b作为区间参数，args为xp.asarray(p)
        res = _chandrupatla_root(self.f, a, b, args=(xp.asarray(p),))
        # 使用stats.norm().ppf(p)计算ppf，并将结果转换为xp数组，赋给ref
        ref = xp.asarray(stats.norm().ppf(p), dtype=xp.asarray(p).dtype)
        # 断言res.x与ref在数值上接近
        xp_assert_close(res.x, ref)

    # 使用pytest装饰器标记，参数化测试用例shape，分别使用空元组、(12,)、(3, 4)、(3, 2, 2)作为参数
    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    # 定义一个测试方法test_flags，接受参数shape和xp
    def test_flags(self, shape, xp):
        # 定义内部函数f，接受xs和js作为参数
        def f(xs, js):
            # 断言js的数据类型为xp.int64
            assert js.dtype == xp.int64
            # 如果使用CuPy作为后端，使用特定的函数列表funcs，根据int(j)选择相应的函数操作xs和js
            if is_cupy(xp):
                funcs = [lambda x: x - 2.5,
                         lambda x: x - 10,
                         lambda x: (x - 0.1)**3,
                         lambda x: xp.full_like(x, xp.nan)]
                return [funcs[int(j)](x) for x, j in zip(xs, js)]

            # 对于其他后端，使用一般的函数列表funcs，根据j选择相应的函数操作xs
            funcs = [lambda x: x - 2.5,
                     lambda x: x - 10,
                     lambda x: (x - 0.1) ** 3,
                     lambda x: xp.nan]
            return [funcs[j](x) for x, j in zip(xs, js)]

        # 定义args为xp.arange(4, dtype=xp.int64)
        args = (xp.arange(4, dtype=xp.int64),)
        # 变量a和b分别被转换为xp数组，并初始化为长度为4的0数组和长度为4的π数组
        a, b = xp.asarray([0.]*4), xp.asarray([xp.pi]*4)
        # 调用_chandrupatla_root函数，传入f作为函数参数，a和b作为区间参数，args=args，maxiter=2
        res = _chandrupatla_root(f, a, b, args=args, maxiter=2)

        # 定义参考标志ref_flags为包含四种特定值的xp数组，数据类型为xp.int32
        ref_flags = xp.asarray([eim._ECONVERGED,
                                eim._ESIGNERR,
                                eim._ECONVERR,
                                eim._EVALUEERR], dtype=xp.int32)
        # 断言res.status与ref_flags相等
        xp_assert_equal(res.status, ref_flags)
    def test_convergence(self, xp):
        # Test that the convergence tolerances behave as expected
        
        # 使用指定种子生成随机数生成器
        rng = np.random.default_rng(2585255913088665241)
        
        # 使用随机数生成器生成长度为3的随机数组，将其转换为xp数组
        p = xp.asarray(rng.random(size=3))
        
        # 定义一个包含两个元素的元组，作为bracket参数
        bracket = (-xp.asarray(5.), xp.asarray(5.))
        
        # 将参数p封装成元组
        args = (p,)
        
        # 定义初始的关键字参数字典kwargs0
        kwargs0 = dict(args=args, xatol=0, xrtol=0, fatol=0, frtol=0)

        # 复制kwargs0，修改xatol为1e-3，调用_chandrupatla_root函数并返回结果res1
        kwargs = kwargs0.copy()
        kwargs['xatol'] = 1e-3
        res1 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查res1.xr - res1.xl是否小于xp数组中各元素都为1e-3的全1数组
        xp_assert_less(res1.xr - res1.xl, xp.full_like(p, 1e-3))
        
        # 修改xatol为1e-6，再次调用_chandrupatla_root函数并返回结果res2
        kwargs['xatol'] = 1e-6
        res2 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查res2.xr - res2.xl是否小于xp数组中各元素都为1e-6的全1数组
        xp_assert_less(res2.xr - res2.xl, xp.full_like(p, 1e-6))
        
        # 使用xp_assert_less断言函数检查res2.xr - res2.xl是否小于res1.xr - res1.xl
        xp_assert_less(res2.xr - res2.xl, res1.xr - res1.xl)

        # 复制kwargs0，修改xrtol为1e-3，调用_chandrupatla_root函数并返回结果res1
        kwargs = kwargs0.copy()
        kwargs['xrtol'] = 1e-3
        res1 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查res1.xr - res1.xl是否小于1e-3 * xp.abs(res1.x)
        xp_assert_less(res1.xr - res1.xl, 1e-3 * xp.abs(res1.x))
        
        # 修改xrtol为1e-6，再次调用_chandrupatla_root函数并返回结果res2
        kwargs['xrtol'] = 1e-6
        res2 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查res2.xr - res2.xl是否小于1e-6 * xp.abs(res2.x)
        xp_assert_less(res2.xr - res2.xl, 1e-6 * xp.abs(res2.x))
        
        # 使用xp_assert_less断言函数检查res2.xr - res2.xl是否小于res1.xr - res1.xl
        xp_assert_less(res2.xr - res2.xl, res1.xr - res1.xl)

        # 复制kwargs0，修改fatol为1e-3，调用_chandrupatla_root函数并返回结果res1
        kwargs = kwargs0.copy()
        kwargs['fatol'] = 1e-3
        res1 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查xp.abs(res1.fun)是否小于xp数组中各元素都为1e-3的全1数组
        xp_assert_less(xp.abs(res1.fun), xp.full_like(p, 1e-3))
        
        # 修改fatol为1e-6，再次调用_chandrupatla_root函数并返回结果res2
        kwargs['fatol'] = 1e-6
        res2 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查xp.abs(res2.fun)是否小于xp数组中各元素都为1e-6的全1数组
        xp_assert_less(xp.abs(res2.fun), xp.full_like(p, 1e-6))
        
        # 使用xp_assert_less断言函数检查xp.abs(res2.fun)是否小于xp.abs(res1.fun)
        xp_assert_less(xp.abs(res2.fun), xp.abs(res1.fun))

        # 复制kwargs0，修改frtol为1e-3，计算x1和x2，定义f0为xp_minimum(xp.abs(self.f(x1, *args)), xp.abs(self.f(x2, *args)))
        kwargs = kwargs0.copy()
        kwargs['frtol'] = 1e-3
        x1, x2 = bracket
        f0 = xp_minimum(xp.abs(self.f(x1, *args)), xp.abs(self.f(x2, *args)))
        res1 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查xp.abs(res1.fun)是否小于1e-3*f0
        xp_assert_less(xp.abs(res1.fun), 1e-3*f0)
        
        # 修改frtol为1e-6，再次调用_chandrupatla_root函数并返回结果res2
        kwargs['frtol'] = 1e-6
        res2 = _chandrupatla_root(self.f, *bracket, **kwargs)
        
        # 使用xp_assert_less断言函数检查xp.abs(res2.fun)是否小于1e-6*f0
        xp_assert_less(xp.abs(res2.fun), 1e-6*f0)
        
        # 使用xp_assert_less断言函数检查xp.abs(res2.fun)是否小于xp.abs(res1.fun)
        xp_assert_less(xp.abs(res2.fun), xp.abs(res1.fun))
    # 定义一个测试方法，用于测试 `maxiter` 参数和 `callback` 接口的行为
    def test_maxiter_callback(self, xp):
        # 使用 xp.asarray 将数值转换为适当的数组形式
        p = xp.asarray(0.612814)
        # 定义一个包含左右边界的元组，作为初始搜索区间
        bracket = (xp.asarray(-5.), xp.asarray(5.))
        # 设定最大迭代次数
        maxiter = 5

        # 定义一个函数 f，其返回值是 special.ndtr(q) - p
        def f(q, p):
            res = special.ndtr(q) - p
            # 设置 f 函数的属性 x 和 fun
            f.x = q
            f.fun = res
            return res
        f.x = None
        f.fun = None

        # 调用 _chandrupatla_root 函数进行根查找，使用给定的 bracket 和 p，最大迭代次数为 maxiter
        res = _chandrupatla_root(f, *bracket, args=(p,), maxiter=maxiter)
        # 断言结果的 success 属性中没有任何 True 值
        assert not xp.any(res.success)
        # 断言结果的 nfev 属性中所有值均为 maxiter+2
        assert xp.all(res.nfev == maxiter+2)
        # 断言结果的 nit 属性中所有值均为 maxiter
        assert xp.all(res.nit == maxiter)

        # 定义一个回调函数 callback，用于在每次迭代时进行检查和断言
        def callback(res):
            # 增加迭代次数计数
            callback.iter += 1
            # 存储当前迭代的结果
            callback.res = res
            # 断言结果对象 res 具有 'x' 属性
            assert hasattr(res, 'x')
            if callback.iter == 0:
                # 在第一次调用时，检查初始搜索区间是否正确
                assert (res.xl, res.xr) == bracket
            else:
                # 对于后续迭代，检查搜索区间是否有变化
                changed = (((res.xl == callback.xl) & (res.xr != callback.xr))
                           | ((res.xl != callback.xl) & (res.xr == callback.xr)))
                assert xp.all(changed)

            # 更新 callback 对象中的 xl 和 xr 属性
            callback.xl = res.xl
            callback.xr = res.xr
            # 断言结果的状态为 _EINPROGRESS，表示搜索仍在进行中
            assert res.status == eim._EINPROGRESS
            # 使用 xp_assert_equal 断言特定值的相等性
            xp_assert_equal(self.f(res.xl, p), res.fl)
            xp_assert_equal(self.f(res.xr, p), res.fr)
            xp_assert_equal(self.f(res.x, p), res.fun)
            if callback.iter == maxiter:
                # 当达到最大迭代次数时，抛出 StopIteration 异常终止迭代
                raise StopIteration
        callback.iter = -1  # 在第一次迭代之前，回调函数被调用一次
        callback.res = None
        callback.xl = None
        callback.xr = None

        # 再次调用 _chandrupatla_root 函数，但这次使用 callback 参数传递定义的回调函数
        res2 = _chandrupatla_root(f, *bracket, args=(p,), callback=callback)

        # 使用循环遍历 res 和 res2 的键，确保它们在终止条件上相同（除了状态）
        for key in res.keys():
            if key == 'status':
                # 对于 'status' 键，分别断言其值与预期值相等
                xp_assert_equal(res[key], xp.asarray(eim._ECONVERR, dtype=xp.int32))
                xp_assert_equal(res2[key], xp.asarray(eim._ECALLBACK, dtype=xp.int32))
            elif key.startswith('_'):
                # 忽略以 '_' 开头的键
                continue
            else:
                # 对于其它键，断言 res2 中的值与 res 中相等
                xp_assert_equal(res2[key], res[key])

    @pytest.mark.parametrize('case', _CHANDRUPATLA_TESTS)
    def test_nit_expected(self, case, xp):
        # 测试 `_chandrupatla` 是否实现了 Chandrupatla 的算法：
        # 在所有 40 个测试用例中，迭代的次数与原始论文中报告的次数相匹配。

        f, bracket, root, nfeval, id = case
        # 设置 Chandrupatla 准则为 abs(x2-x1) < abs(xmin)*xrtol + xatol，
        # 在测试中我们使用比 Chandrupatla 更标准的 xrtol = 4e-10。
        bracket = (xp.asarray(bracket[0], dtype=xp.float64),
                   xp.asarray(bracket[1], dtype=xp.float64))
        root = xp.asarray(root, dtype=xp.float64)

        # 调用 `_chandrupatla_root` 函数进行根查找
        res = _chandrupatla_root(f, *bracket, xrtol=4e-10, xatol=1e-5)
        
        # 断言函数值接近于给定根
        xp_assert_close(res.fun, xp.asarray(f(root), dtype=xp.float64),
                        rtol=1e-8, atol=2e-3)
        
        # 断言迭代次数与预期值相等
        xp_assert_equal(res.nfev, xp.asarray(nfeval, dtype=xp.int32))

    @pytest.mark.parametrize("root", (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize("dtype", ('float16', 'float32', 'float64'))
    def test_dtype(self, root, dtype, xp):
        # 测试数据类型是否被保留

        not_numpy = not is_numpy(xp)
        if not_numpy and dtype == 'float16':
            # 如果不是 NumPy 并且数据类型是 'float16'，则跳过测试
            pytest.skip("`float16` dtype only supported for NumPy arrays.")

        dtype = getattr(xp, dtype, None)
        if dtype is None:
            # 如果数据类型不被支持，则跳过测试
            pytest.skip(f"{xp} does not support {dtype}")

        def f(x, root):
            res = (x - root) ** 3.
            if is_numpy(xp):  # NumPy 不能保留数据类型
                return xp.asarray(res, dtype=dtype)
            return res

        # 初始化数组边界和根
        a, b = xp.asarray(-3, dtype=dtype), xp.asarray(3, dtype=dtype)
        root = xp.asarray(root, dtype=dtype)

        # 使用 `_chandrupatla_root` 函数寻找根
        res = _chandrupatla_root(f, a, b, args=(root,), xatol=1e-3)
        
        # 尝试断言结果的根接近于预期根
        try:
            xp_assert_close(res.x, root, atol=1e-3)
        except AssertionError:
            assert res.x.dtype == dtype
            # 如果断言失败，确保所有函数值为 0
            xp.all(res.fun == 0)
    # 定义单元测试方法，用于测试输入验证，检查是否能正确处理错误消息

    def test_input_validation(self, xp):
        # Test input validation for appropriate error messages
        
        # 定义一个简单的函数 func(x)，返回输入的 x
        def func(x):
            return x
        
        # 准备错误消息，验证在 func 不可调用时是否会引发 ValueError 异常
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            # 准备用于测试的区间 bracket，并调用 _chandrupatla_root 函数
            bracket = xp.asarray(-4), xp.asarray(4)
            _chandrupatla_root(None, *bracket)

        # 准备错误消息，验证在 abscissae 和函数输出不是实数时是否会引发 ValueError 异常
        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            bracket = xp.asarray(-4+1j), xp.asarray(4)
            _chandrupatla_root(func, *bracket)

        # 准备错误消息，验证在广播问题时是否会引发 ValueError 或 RuntimeError 异常
        message = "...not be broadcast..."  # 所有相关错误消息包含此部分
        with pytest.raises((ValueError, RuntimeError), match=message):
            bracket = xp.asarray([-2, -3]), xp.asarray([3, 4, 5])
            _chandrupatla_root(func, *bracket)

        # 准备错误消息，验证函数返回的数组形状不符合预期时是否会引发 ValueError 异常
        message = "The shape of the array returned by `func`..."
        with pytest.raises(ValueError, match=message):
            bracket = xp.asarray([-3, -3]), xp.asarray([5, 5])
            _chandrupatla_root(lambda x: [x[0], x[1], x[1]], *bracket)

        # 准备错误消息，验证在容差参数不合理时是否会引发 ValueError 异常
        message = 'Tolerances must be non-negative scalars.'
        bracket = xp.asarray(-4), xp.asarray(4)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, xatol=-1)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, xrtol=xp.nan)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, fatol='ekki')
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, frtol=xp.nan)

        # 准备错误消息，验证在最大迭代次数不合理时是否会引发 ValueError 异常
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, maxiter=-1)

        # 准备错误消息，验证在回调函数不可调用时是否会引发 ValueError 异常
        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_root(func, *bracket, callback='shrubbery')
```