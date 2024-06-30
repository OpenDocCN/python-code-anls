# `D:\src\scipysrc\scipy\scipy\stats\tests\test_distributions.py`

```
"""
Test functions for stats module
"""
# 导入警告模块
import warnings
# 导入正则表达式模块
import re
# 导入系统模块
import sys
# 导入pickle模块
import pickle
# 从路径模块中导入Path类
from pathlib import Path
# 导入操作系统功能模块
import os
# 导入JSON模块
import json
# 导入平台信息模块
import platform

# 导入numpy测试模块中的各种断言方法
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_, assert_warns,
                           assert_array_less, suppress_warnings,
                           assert_array_max_ulp, IS_PYPY)
# 导入pytest测试框架及其异常处理模块
import pytest
from pytest import raises as assert_raises

# 导入numpy库和特定函数
import numpy as np
from numpy import typecodes, array
# 从numpy库中导入记录数组相关的函数
from numpy.lib.recfunctions import rec_append_fields
# 导入scipy的特殊函数模块
from scipy import special
# 导入随机数状态检查函数
from scipy._lib._util import check_random_state
# 导入积分相关模块
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
                             cumulative_trapezoid)
# 导入scipy统计模块
import scipy.stats as stats
# 导入分布参数相关模块
from scipy.stats._distn_infrastructure import argsreduce
# 导入scipy统计分布模块
import scipy.stats.distributions

# 导入scipy特殊函数模块
from scipy.special import xlogy, polygamma, entr
# 导入统计分布参数定义模块
from scipy.stats._distr_params import distcont, invdistcont
# 从测试模块中导入离散基础测试相关模块
from .test_discrete_basic import distdiscrete, invdistdiscrete
# 导入连续分布模块
from scipy.stats._continuous_distns import FitDataError, _argus_phi
# 导入优化算法相关模块
from scipy.optimize import root, fmin, differential_evolution
# 导入迭代工具模块
from itertools import product

# 设置优化标志，用于指示是否移除文档字符串
DOCSTRINGS_STRIPPED = sys.flags.optimize > 1

# 在macOS 11和Intel CPU上失败。见gh-14901
MACOS_INTEL = (sys.platform == 'darwin') and (platform.machine() == 'x86_64')

# 在测试支持方法修复时跳过的分布列表
skip_test_support_gh13294_regression = ['tukeylambda', 'pearson3']


def _assert_hasattr(a, b, msg=None):
    """
    Asserts that object 'a' has attribute 'b'.
    
    Parameters:
    a : object
        The object to check for the attribute.
    b : str
        The name of the attribute to check.
    msg : str, optional
        Custom message to display on assertion failure.
    """
    if msg is None:
        msg = f'{a} does not have attribute {b}'
    assert_(hasattr(a, b), msg=msg)


def test_api_regression():
    """
    Test case for API regression related to scipy.stats.distributions.f_gen.
    """
    _assert_hasattr(scipy.stats.distributions, 'f_gen')


def test_distributions_submodule():
    """
    Test case for the distributions submodule of scipy.stats.
    """
    # 获取实际的分布名称集合
    actual = set(scipy.stats.distributions.__all__)
    # 获取连续分布名称列表
    continuous = [dist[0] for dist in distcont]    # continuous dist names
    # 获取离散分布名称列表
    discrete = [dist[0] for dist in distdiscrete]  # discrete dist names
    # 其他常规分布名称列表
    other = ['rv_discrete', 'rv_continuous', 'rv_histogram',
             'entropy', 'trapz']
    # 预期的分布名称集合为连续分布、离散分布及其他常规分布的结合
    expected = continuous + discrete + other

    # 需要移除以“<”开头的字符串，例如:
    # <scipy.stats._continuous_distns.trapezoid_gen at 0x1df83bbc688>
    expected = set(filter(lambda s: not str(s).startswith('<'), expected))

    # 断言实际的分布名称集合与预期的分布名称集合相等
    assert actual == expected


class TestVonMises:
    """
    Test class for Von Mises distribution.
    """
    @pytest.mark.parametrize('k', [0.1, 1, 101])
    @pytest.mark.parametrize('x', [0, 1, np.pi, 10, 100])
    # 定义一个测试函数，用于验证周期性 von Mises 分布的概率密度函数
    def test_vonmises_periodic(self, k, x):
        # 定义内部函数，验证给定参数下的 von Mises 分布的概率密度函数
        def check_vonmises_pdf_periodic(k, L, s, x):
            # 创建 von Mises 分布对象，设置参数 k, L, s
            vm = stats.vonmises(k, loc=L, scale=s)
            # 断言：验证原点为 L，标度为 s 的 von Mises 分布在 x 和 x % (2 * np.pi * s) 处的概率密度函数值近似相等
            assert_almost_equal(vm.pdf(x), vm.pdf(x % (2 * np.pi * s)))

        # 定义内部函数，验证给定参数下的 von Mises 分布的累积分布函数
        def check_vonmises_cdf_periodic(k, L, s, x):
            # 创建 von Mises 分布对象，设置参数 k, L, s
            vm = stats.vonmises(k, loc=L, scale=s)
            # 断言：验证原点为 L，标度为 s 的 von Mises 分布在 x 和 x % (2 * np.pi * s) 处的累积分布函数的整数部分相等
            assert_almost_equal(vm.cdf(x) % 1,
                                vm.cdf(x % (2 * np.pi * s)) % 1)

        # 分别使用不同参数进行周期性 von Mises 分布的概率密度函数验证
        check_vonmises_pdf_periodic(k, 0, 1, x)
        check_vonmises_pdf_periodic(k, 1, 1, x)
        check_vonmises_pdf_periodic(k, 0, 10, x)

        # 分别使用不同参数进行周期性 von Mises 分布的累积分布函数验证
        check_vonmises_cdf_periodic(k, 0, 1, x)
        check_vonmises_cdf_periodic(k, 1, 1, x)
        check_vonmises_cdf_periodic(k, 0, 10, x)

    # 定义一个测试函数，用于验证 von Mises 分布在直线支持上的边界
    def test_vonmises_line_support(self):
        # 断言：验证 von Mises 分布在直线支持上的左边界
        assert_equal(stats.vonmises_line.a, -np.pi)
        # 断言：验证 von Mises 分布在直线支持上的右边界
        assert_equal(stats.vonmises_line.b, np.pi)

    # 定义一个测试函数，用于验证 von Mises 分布的数值计算
    def test_vonmises_numerical(self):
        # 创建 von Mises 分布对象，设置参数 800
        vm = stats.vonmises(800)
        # 断言：验证 von Mises 分布在 x=0 处的累积分布函数值近似为 0.5
        assert_almost_equal(vm.cdf(0), 0.5)

    # 预期的 von Mises 概率密度函数值使用 mpmath 精确计算（50 位精度）：
    #
    # def vmpdf_mp(x, kappa):
    #     x = mpmath.mpf(x)
    #     kappa = mpmath.mpf(kappa)
    #     num = mpmath.exp(kappa*mpmath.cos(x))
    #     den = 2 * mpmath.pi * mpmath.besseli(0, kappa)
    #     return num/den

    # 使用 pytest 的参数化标记，验证 von Mises 分布的概率密度函数
    @pytest.mark.parametrize('x, kappa, expected_pdf',
                             [(0.1, 0.01, 0.16074242744907072),
                              (0.1, 25.0, 1.7515464099118245),
                              (0.1, 800, 0.2073272544458798),
                              (2.0, 0.01, 0.15849003875385817),
                              (2.0, 25.0, 8.356882934278192e-16),
                              (2.0, 800, 0.0)])
    # 定义一个测试函数，用于验证 von Mises 分布的概率密度函数计算
    def test_vonmises_pdf(self, x, kappa, expected_pdf):
        # 计算 von Mises 分布在给定 x 和 kappa 下的概率密度函数值
        pdf = stats.vonmises.pdf(x, kappa)
        # 断言：验证计算得到的概率密度函数值与预期值的近似程度
        assert_allclose(pdf, expected_pdf, rtol=1e-15)

    # 预期的 von Mises 熵使用 mpmath 精确计算（50 位精度）：
    #
    # def vonmises_entropy(kappa):
    #     kappa = mpmath.mpf(kappa)
    #     return (-kappa * mpmath.besseli(1, kappa) /
    #             mpmath.besseli(0, kappa) + mpmath.log(2 * mpmath.pi *
    #             mpmath.besseli(0, kappa)))
    # >>> float(vonmises_entropy(kappa))

    # 使用 pytest 的参数化标记，验证 von Mises 分布的熵计算
    @pytest.mark.parametrize('kappa, expected_entropy',
                             [(1, 1.6274014590199897),
                              (5, 0.6756431570114528),
                              (100, -0.8811275441649473),
                              (1000, -2.03468891852547),
                              (2000, -2.3813876496587847)])
    # 定义一个测试函数，用于验证 von Mises 分布的熵计算
    def test_vonmises_entropy(self, kappa, expected_entropy):
        # 计算 von Mises 分布在给定 kappa 下的熵
        entropy = stats.vonmises.entropy(kappa)
        # 断言：验证计算得到的熵与预期值的近似程度
        assert_allclose(entropy, expected_entropy, rtol=1e-13)
    # 定义一个单元测试函数，用于测试 vonmises 分布中随机变量环绕的情况，参考了问题 gh-4598 的讨论
    def test_vonmises_rvs_gh4598(self):
        # 设置随机数种子
        seed = 30899520
        # 创建三个不同的随机数生成器实例，使用相同的种子保证结果可复现性
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        rng3 = np.random.default_rng(seed)
        # 使用第一个随机数生成器生成 vonmises 分布的随机变量
        rvs1 = stats.vonmises(1, loc=0, scale=1).rvs(random_state=rng1)
        # 使用第二个随机数生成器生成 vonmises 分布的随机变量，设置 loc=2*pi
        rvs2 = stats.vonmises(1, loc=2*np.pi, scale=1).rvs(random_state=rng2)
        # 使用第三个随机数生成器生成 vonmises 分布的随机变量，动态设置 scale
        rvs3 = stats.vonmises(1, loc=0,
                              scale=(2*np.pi/abs(rvs1)+1)).rvs(random_state=rng3)
        # 断言生成的随机变量在数值上接近，使用绝对容差 1e-15
        assert_allclose(rvs1, rvs2, atol=1e-15)
        assert_allclose(rvs1, rvs3, atol=1e-15)

    # 使用 Wolfram Alpha 计算的 vonmises 分布的 LOGPDF 预期值
    @pytest.mark.parametrize('x, kappa, expected_logpdf',
                             [(0.1, 0.01, -1.8279520246003170),
                              (0.1, 25.0, 0.5604990605420549),
                              (0.1, 800, -1.5734567947337514),
                              (2.0, 0.01, -1.8420635346185686),
                              (2.0, 25.0, -34.7182759850871489),
                              (2.0, 800, -1130.4942582548682739)])
    # 定义一个单元测试函数，用于测试 vonmises 分布的 logpdf 方法
    def test_vonmises_logpdf(self, x, kappa, expected_logpdf):
        # 计算 vonmises 分布在给定参数下的 logpdf 值
        logpdf = stats.vonmises.logpdf(x, kappa)
        # 断言计算得到的 logpdf 值与预期值接近，使用相对容差 1e-15
        assert_allclose(logpdf, expected_logpdf, rtol=1e-15)

    # 定义一个单元测试函数，用于测试 vonmises 分布的期望值计算
    def test_vonmises_expect(self):
        """
        Test that the vonmises expectation values are
        computed correctly.  This test checks that the
        numeric integration estimates the correct normalization
        (1) and mean angle (loc).  These expectations are
        independent of the chosen 2pi interval.
        """
        # 创建一个随机数生成器实例
        rng = np.random.default_rng(6762668991392531563)

        # 从随机数生成器中生成 loc, kappa, lb 三个随机数值
        loc, kappa, lb = rng.random(3) * 10
        # 使用 vonmises 分布计算期望值，验证其是否接近 1
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)

        # 定义边界范围
        bounds = lb, lb + 2 * np.pi
        # 使用 vonmises 分布计算期望值，验证其是否接近 1，包含边界参数
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1, *bounds)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)

        # 使用 vonmises 分布计算期望值，验证其角度部分是否正确，包含复数函数参数
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: np.exp(1j*x),
                                                          *bounds, complex_func=1)
        assert_allclose(np.angle(res), loc % (2*np.pi))
        assert np.issubdtype(res.dtype, np.complexfloating)

    # 标记这个测试函数为慢速测试，并且使用多个参数化来进行测试
    @pytest.mark.xslow
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_shape", [1, 100, 1e8])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_shape', [True, False])
    # 定义测试函数，用于测试最大似然估计的优化器是否正常工作
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_shape,
                                    fix_loc, fix_shape):
        # 如果固定位置和形状均为真，则跳过测试，并给出相应信息
        if fix_shape and fix_loc:
            pytest.skip("Nothing to fit.")

        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(6762668991392531563)
        # 生成符合冯·米塞斯分布的随机数据
        data = stats.vonmises.rvs(rvs_shape, size=1000, loc=rvs_loc,
                                  random_state=rng)

        # 初始化参数字典，设置默认的尺度因子为1
        kwds = {'fscale': 1}
        # 如果固定位置为真，则将随机位置添加到参数字典中
        if fix_loc:
            kwds['floc'] = rvs_loc
        # 如果固定形状为真，则将随机形状参数添加到参数字典中
        if fix_shape:
            kwds['f0'] = rvs_shape

        # 调用自定义的断言函数，验证最大似然估计的对数似然是否符合预期
        _assert_less_or_close_loglike(stats.vonmises, data,
                                      stats.vonmises.nnlf, **kwds)

    # 使用pytest标记为慢速测试的函数，用于测试冯·米塞斯分布的位置参数异常情况
    @pytest.mark.slow
    def test_vonmises_fit_bad_floc(self):
        # 给定一组特定的测试数据
        data = [-0.92923506, -0.32498224, 0.13054989, -0.97252014, 2.79658071,
                -0.89110948, 1.22520295, 1.44398065, 2.49163859, 1.50315096,
                3.05437696, -2.73126329, -3.06272048, 1.64647173, 1.94509247,
                -1.14328023, 0.8499056, 2.36714682, -1.6823179, -0.88359996]
        # 将数据转换为NumPy数组
        data = np.asarray(data)
        # 设置位置参数的参考值为负半圆周率
        loc = -0.5 * np.pi
        # 使用给定的位置参数进行最大似然估计，获取估计的卡帕、位置和尺度
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
        # 断言估计的卡帕参数应接近浮点数的最小值
        assert kappa_fit == np.finfo(float).tiny
        # 调用自定义的断言函数，验证最大似然估计的对数似然是否符合预期
        _assert_less_or_close_loglike(stats.vonmises, data,
                                      stats.vonmises.nnlf, fscale=1, floc=loc)

    # 使用pytest参数化标记的测试函数，用于测试冯·米塞斯分布的未包装数据
    @pytest.mark.parametrize('sign', [-1, 1])
    def test_vonmises_fit_unwrapped_data(self, sign):
        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(6762668991392531563)
        # 生成冯·米塞斯分布的随机数据，位置参数为正负半圆周率乘以0.5，集中在卡帕为10的分布上
        data = stats.vonmises(loc=sign*0.5*np.pi, kappa=10).rvs(100000,
                                                                random_state=rng)
        # 将数据进行平移，使其不再集中在0附近
        shifted_data = data + 4*np.pi
        # 使用原始数据进行最大似然估计，获取估计的卡帕、位置和尺度
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
        # 使用平移后的数据进行最大似然估计，获取估计的卡帕、位置和尺度
        kappa_fit_shifted, loc_fit_shifted, _ = stats.vonmises.fit(shifted_data)
        # 断言位置参数的估计值应接近
        assert_allclose(loc_fit, loc_fit_shifted)
        # 断言卡帕参数的估计值应接近
        assert_allclose(kappa_fit, kappa_fit_shifted)
        # 断言尺度参数的估计值为1
        assert scale_fit == 1
        # 断言位置参数的估计值在-pi和pi之间
        assert -np.pi < loc_fit < np.pi

    # 测试冯·米塞斯分布的卡帕参数为0的情况
    def test_vonmises_kappa_0_gh18166(self):
        # 创建一个冯·米塞斯分布对象，卡帕参数设为0
        dist = stats.vonmises(0)
        # 断言PDF在0处的值应接近1 / (2 * pi)
        assert_allclose(dist.pdf(0), 1 / (2 * np.pi), rtol=1e-15)
        # 断言CDF在pi/2处的值应接近0.75
        assert_allclose(dist.cdf(np.pi/2), 0.75, rtol=1e-15)
        # 断言SF在-pi/2处的值应接近0.75
        assert_allclose(dist.sf(-np.pi/2), 0.75, rtol=1e-15)
        # 断言PPF在0.9处的值应接近pi * 0.8
        assert_allclose(dist.ppf(0.9), np.pi*0.8, rtol=1e-15)
        # 断言均值应接近0
        assert_allclose(dist.mean(), 0, atol=1e-15)
        # 断言期望值应接近0
        assert_allclose(dist.expect(), 0, atol=1e-15)
        # 断言生成的随机样本的绝对值均不大于pi
        assert np.all(np.abs(dist.rvs(size=10, random_state=1234)) <= np.pi)

    # 测试所有数据相等时的冯·米塞斯分布估计
    def test_vonmises_fit_equal_data(self):
        # 当所有数据相等时，预期卡帕参数应为1e16
        kappa, loc, scale = stats.vonmises.fit([0])
        assert kappa == 1e16 and loc == 0 and scale == 1
    # 定义一个测试方法，用于测试 Von Mises 分布拟合的边界情况处理。
    def test_vonmises_fit_bounds(self):
        # 对于某些输入数据，数值上会违反根范围。
        # 测试确保这种情况被处理。以下输入数据被精心设计以触发当前选择的边界和特定方式计算边界和目标函数的边界违规。

        # 测试当下限违反时不会引发异常。
        scipy.stats.vonmises.fit([0, 3.7e-08], floc=0)

        # 测试当上限违反时不会引发异常。
        scipy.stats.vonmises.fit([np.pi/2*(1-4.86e-9)], floc=0)
# 定义一个内部函数，用于验证通过 dist.fit() 计算出的负对数似然函数值是否小于或等于通过通用拟合方法计算的结果。
# 如果未提供 func 参数，则默认使用 dist.nnlf 作为负对数似然函数。
def _assert_less_or_close_loglike(dist, data, func=None, maybe_identical=False,
                                  **kwds):
    """
    This utility function checks that the negative log-likelihood function
    (or `func`) of the result computed using dist.fit() is less than or equal
    to the result computed using the generic fit method.  Because of
    normal numerical imprecision, the "equality" check is made using
    `np.allclose` with a relative tolerance of 1e-15.
    """
    # 如果未提供 func 参数，则使用 dist.nnlf 作为负对数似然函数
    if func is None:
        func = dist.nnlf

    # 使用 dist.fit() 对数据进行拟合，返回分析解的最大似然估计
    mle_analytical = dist.fit(data, **kwds)
    # 使用超类方法调用通用拟合方法，返回数值优化估计
    numerical_opt = super(type(dist), dist).fit(data, **kwds)

    # 对分析解的最大似然估计和数值优化估计进行断言，预期它们不完全相同
    if not maybe_identical:
        assert np.any(mle_analytical != numerical_opt)

    # 计算分析解和数值优化估计的负对数似然函数值
    ll_mle_analytical = func(mle_analytical, data)
    ll_numerical_opt = func(numerical_opt, data)
    
    # 断言分析解的负对数似然函数值小于或等于数值优化估计的值，或者它们在数值上非常接近
    assert (ll_mle_analytical <= ll_numerical_opt or
            np.allclose(ll_mle_analytical, ll_numerical_opt, rtol=1e-15))

    # 理想情况下，应该检查形状是否被正确固定，但由于固定方式多样（如 f0, fix_a, fa），这变得复杂。
    if 'floc' in kwds:
        assert mle_analytical[-2] == kwds['floc']
    if 'fscale' in kwds:
        assert mle_analytical[-1] == kwds['fscale']


# 定义一个测试函数，用于验证在特定情况下拟合函数的警告是否如预期抛出
def assert_fit_warnings(dist):
    param = ['floc', 'fscale']
    # 如果 dist 具有形状参数，则根据其数量扩展 param 列表
    if dist.shapes:
        nshapes = len(dist.shapes.split(","))
        param += ['f0', 'f1', 'f2'][:nshapes]
    # 创建一个包含非有限值的数据列表
    data = [1, 2, 3]
    
    # 使用 pytest 检查以下情况是否抛出 Runtime 错误，并匹配特定的错误信息
    with pytest.raises(RuntimeError,
                       match="All parameters fixed. There is nothing "
                       "to optimize."):
        dist.fit(data, **all_fixed)
    
    # 使用 pytest 检查以下情况是否抛出 Value 错误，并匹配特定的错误信息
    with pytest.raises(ValueError,
                       match="The data contains non-finite values"):
        dist.fit([np.nan])
    with pytest.raises(ValueError,
                       match="The data contains non-finite values"):
        dist.fit([np.inf])
    
    # 使用 pytest 检查以下情况是否抛出 Type 错误，并匹配特定的错误信息
    with pytest.raises(TypeError, match="Unknown keyword arguments:"):
        dist.fit(data, extra_keyword=2)
    with pytest.raises(TypeError, match="Too many positional arguments."):
        dist.fit(data, *[1]*(len(param) - 1))


# 使用 pytest 的参数化测试装饰器，对不同的分布进行测试
@pytest.mark.parametrize('dist',
                         ['alpha', 'betaprime',
                          'fatiguelife', 'invgamma', 'invgauss', 'invweibull',
                          'johnsonsb', 'levy', 'levy_l', 'lognorm', 'gibrat',
                          'powerlognorm', 'rayleigh', 'wald'])
def test_support(dist):
    """gh-6235"""
    # 将 distcont 中的分布名称映射到 args 字典中
    dct = dict(distcont)
    args = dct[dist]

    # 根据分布名称获取对应的概率密度函数，并进行以下断言
    dist = getattr(stats, dist)

    # 断言分布在下界 dist.a 处的概率密度函数值接近 0
    assert_almost_equal(dist.pdf(dist.a, *args), 0)
    # 断言分布在下界 dist.a 处的对数概率密度函数值为负无穷
    assert_equal(dist.logpdf(dist.a, *args), -np.inf)
    # 断言分布在上界 dist.b 处的概率密度函数值接近 0
    assert_almost_equal(dist.pdf(dist.b, *args), 0)
    # 使用断言检查 dist.b 在给定参数 args 下的对数概率密度函数值是否为负无穷
    assert_equal(dist.logpdf(dist.b, *args), -np.inf)
class TestRandInt:
    # 设置每次生成随机数的种子，以便结果可复现
    def setup_method(self):
        np.random.seed(1234)

    # 测试 stats.randint.rvs 方法
    def test_rvs(self):
        # 生成指定范围内的随机整数数组
        vals = stats.randint.rvs(5, 30, size=100)
        # 断言所有生成的随机数小于30且大于等于5
        assert_(np.all(vals < 30) & np.all(vals >= 5))
        # 断言生成的数组长度为100
        assert_(len(vals) == 100)
        # 生成指定范围内形状为(2, 50)的随机整数数组
        vals = stats.randint.rvs(5, 30, size=(2, 50))
        # 断言生成的数组形状为(2, 50)
        assert_(np.shape(vals) == (2, 50))
        # 断言生成的数组数据类型是整数类型
        assert_(vals.dtype.char in typecodes['AllInteger'])
        # 生成单个随机整数
        val = stats.randint.rvs(15, 46)
        # 断言生成的随机数大于等于15且小于46
        assert_((val >= 15) & (val < 46))
        # 断言生成的值是 NumPy 标量类型
        assert_(isinstance(val, np.ScalarType), msg=repr(type(val)))
        # 生成 stats.randint 对象后再生成随机数数组
        val = stats.randint(15, 46).rvs(3)
        # 断言生成的数组数据类型是整数类型
        assert_(val.dtype.char in typecodes['AllInteger'])

    # 测试 stats.randint.pmf 方法
    def test_pdf(self):
        # 创建一个包含0到35的数组
        k = np.r_[0:36]
        # 根据条件设置数组元素值
        out = np.where((k >= 5) & (k < 30), 1.0/(30-5), 0)
        # 计算给定范围内的离散随机变量的概率质量函数
        vals = stats.randint.pmf(k, 5, 30)
        # 断言计算得到的概率质量函数与预期输出一致
        assert_array_almost_equal(vals, out)

    # 测试 stats.randint.cdf 方法
    def test_cdf(self):
        # 创建一个0到36的等间隔数组
        x = np.linspace(0, 36, 100)
        # 向下取整得到数组
        k = np.floor(x)
        # 根据条件设置数组元素值
        out = np.select([k >= 30, k >= 5], [1.0, (k-5.0+1)/(30-5.0)], 0)
        # 计算给定范围内的离散随机变量的累积分布函数
        vals = stats.randint.cdf(x, 5, 30)
        # 断言计算得到的累积分布函数与预期输出一致，精确度为12位小数
        assert_array_almost_equal(vals, out, decimal=12)


class TestBinom:
    # 设置每次生成随机数的种子，以便结果可复现
    def setup_method(self):
        np.random.seed(1234)

    # 测试 stats.binom.rvs 方法
    def test_rvs(self):
        # 生成指定参数下的二项分布随机数数组
        vals = stats.binom.rvs(10, 0.75, size=(2, 50))
        # 断言生成的随机数大于等于0且小于等于10
        assert_(np.all(vals >= 0) & np.all(vals <= 10))
        # 断言生成的数组形状为(2, 50)
        assert_(np.shape(vals) == (2, 50))
        # 断言生成的数组数据类型是整数类型
        assert_(vals.dtype.char in typecodes['AllInteger'])
        # 生成指定参数下的二项分布的单个随机数
        val = stats.binom.rvs(10, 0.75)
        # 断言生成的值是整数类型
        assert_(isinstance(val, int))
        # 生成 stats.binom 对象后再生成随机数数组
        val = stats.binom(10, 0.75).rvs(3)
        # 断言生成的数组数据类型是整数类型
        assert_(isinstance(val, np.ndarray))
        # 断言生成的数组数据类型是整数类型
        assert_(val.dtype.char in typecodes['AllInteger'])

    # 测试 stats.binom.pmf 方法
    def test_pmf(self):
        # 回归测试，验证 Ticket #1842
        vals1 = stats.binom.pmf(100, 100, 1)
        vals2 = stats.binom.pmf(0, 100, 0)
        # 断言计算得到的概率质量函数与预期输出一致，相对误差小于等于1e-15
        assert_allclose(vals1, 1.0, rtol=1e-15, atol=0)
        assert_allclose(vals2, 1.0, rtol=1e-15, atol=0)

    # 测试 stats.binom.entropy 方法
    def test_entropy(self):
        # 基本的熵测试
        b = stats.binom(2, 0.5)
        expected_p = np.array([0.25, 0.5, 0.25])
        expected_h = -sum(xlogy(expected_p, expected_p))
        h = b.entropy()
        # 断言计算得到的熵与预期输出一致
        assert_allclose(h, expected_h)

        b = stats.binom(2, 0.0)
        h = b.entropy()
        # 断言熵为0
        assert_equal(h, 0.0)

        b = stats.binom(2, 1.0)
        h = b.entropy()
        # 断言熵为0
        assert_equal(h, 0.0)

    # 测试 stats.binom 对象在 p=0 时不会生成警告
    def test_warns_p0(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # 断言当 p=0 时，二项分布的均值为0
            assert_equal(stats.binom(n=2, p=0).mean(), 0)
            # 断言当 p=0 时，二项分布的标准差为0
            assert_equal(stats.binom(n=2, p=0).std(), 0)

    # 测试 stats.binom.ppf 方法在 p=1 时的情况
    def test_ppf_p1(self):
        n = 4
        # 断言当 p=1 时，累积概率分布函数的百分位点等于 n
        assert stats.binom.ppf(q=0.3, n=n, p=1.0) == n
    # 定义一个测试函数，用于测试泊松分布的概率质量函数（pmf）
    def test_pmf_poisson(self):
        # 检查问题编号 gh-17146 是否已解决：将二项分布（binom）转换为泊松分布（poisson）
        n = 1541096362225563.0
        p = 1.0477878413173978e-18
        x = np.arange(3)
        # 计算二项分布的概率质量函数值
        res = stats.binom.pmf(x, n=n, p=p)
        # 计算泊松分布的概率质量函数值作为参考值
        ref = stats.poisson.pmf(x, n * p)
        # 检查计算结果是否在指定的绝对容差范围内接近
        assert_allclose(res, ref, atol=1e-16)
    
    # 定义另一个测试函数，用于测试二项分布的概率质量函数和累积分布函数之间的关系
    def test_pmf_cdf(self):
        # 检查问题编号 gh-17809 是否已解决：二项分布的概率质量函数（pmf）在 r=0 时是否近似等于累积分布函数（cdf）在 r=0 时的值
        n = 25.0 * 10 ** 21
        p = 1.0 * 10 ** -21
        r = 0
        # 计算二项分布在 r=0 时的概率质量函数值
        res = stats.binom.pmf(r, n, p)
        # 计算二项分布在 r=0 时的累积分布函数值作为参考值
        ref = stats.binom.cdf(r, n, p)
        # 检查计算结果是否在指定的绝对容差范围内接近
        assert_allclose(res, ref, atol=1e-16)
    
    # 定义第三个测试函数，用于测试二项分布在极限情况下（p~1, n~oo）是否会出现除以零警告
    def test_pmf_gh15101(self):
        # 检查问题编号 gh-15101 是否已解决：当 p 接近 1 且 n 接近无穷大时，计算二项分布的概率质量函数是否会出现除以零警告
        res = stats.binom.pmf(3, 2000, 0.999)
        # 检查计算结果是否接近于零，以指定的绝对容差作为阈值
        assert_allclose(res, 0, atol=1e-16)
class TestArcsine:

    def test_endpoints(self):
        # Regression test for gh-13697.  The following calculation
        # should not generate a warning.
        # 计算 arcsine 分布的概率密度函数在 [0, 1] 处的取值，预期结果为无穷大
        p = stats.arcsine.pdf([0, 1])
        assert_equal(p, [np.inf, np.inf])


class TestBernoulli:
    
    def setup_method(self):
        # 设置随机种子以便复现测试结果
        np.random.seed(1234)

    def test_rvs(self):
        # 生成指定参数下的 Bernoulli 分布随机变量，并进行多个断言检查
        vals = stats.bernoulli.rvs(0.75, size=(2, 50))
        assert_(np.all(vals >= 0) & np.all(vals <= 1))
        assert_(np.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        # 生成单个 Bernoulli 分布随机变量，并进行类型检查
        val = stats.bernoulli.rvs(0.75)
        assert_(isinstance(val, int))
        # 生成指定参数下的 Bernoulli 分布随机变量数组，并进行类型和数据类型检查
        val = stats.bernoulli(0.75).rvs(3)
        assert_(isinstance(val, np.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_entropy(self):
        # 对 Bernoulli 分布的熵进行简单测试
        b = stats.bernoulli(0.25)
        expected_h = -0.25*np.log(0.25) - 0.75*np.log(0.75)
        h = b.entropy()
        assert_allclose(h, expected_h)
        
        # 测试 p=0.0 时 Bernoulli 分布的熵
        b = stats.bernoulli(0.0)
        h = b.entropy()
        assert_equal(h, 0.0)
        
        # 测试 p=1.0 时 Bernoulli 分布的熵
        b = stats.bernoulli(1.0)
        h = b.entropy()
        assert_equal(h, 0.0)


class TestBradford:
    # Regression test for gh-6216
    def test_cdf_ppf(self):
        # 设置参数 c 和输入变量 x
        c = 0.1
        x = np.logspace(-20, -4)
        # 计算 Bradford 分布的累积分布函数和其分位点函数，并进行数值近似检查
        q = stats.bradford.cdf(x, c)
        xx = stats.bradford.ppf(q, c)
        assert_allclose(x, xx)


class TestChi:

    # Exact values computed externally for chi.sf(10, 4) and chi.mean(df=1000)
    CHI_SF_10_4 = 9.83662422461598e-21
    CHI_MEAN_1000 = 31.614871896980

    def test_sf(self):
        # 计算 chi 分布的生存函数并进行数值近似检查
        s = stats.chi.sf(10, 4)
        assert_allclose(s, self.CHI_SF_10_4, rtol=1e-15)

    def test_isf(self):
        # 计算 chi 分布的逆生存函数并进行数值近似检查
        x = stats.chi.isf(self.CHI_SF_10_4, 4)
        assert_allclose(x, 10, rtol=1e-15)

    # Reference values for mean were computed via mpmath for specific degrees of freedom
    @pytest.mark.parametrize('df, ref',
                             [(1e3, CHI_MEAN_1000),
                              (1e14, 9999999.999999976)]
                            )
    def test_mean(self, df, ref):
        # 计算 chi 分布的均值并进行数值近似检查
        assert_allclose(stats.chi.mean(df), ref, rtol=1e-12)
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize('df, ref',
                             # 参数化测试的参数列表，包括测试数据和期望结果
                             [(1e-4, -9989.7316027504),
                              (1, 0.7257913526447274),
                              (1e3, 1.0721981095025448),
                              (1e10, 1.0723649429080335),
                              (1e100, 1.0723649429247002)])
    # 定义测试方法 test_entropy，其中 df 和 ref 是参数化测试的输入和期望输出
    def test_entropy(self, df, ref):
        # 使用 assert_allclose 断言函数，验证 stats.chi(df).entropy() 的返回值与 ref 是否在相对误差 1e-15 内相等
        assert_allclose(stats.chi(df).entropy(), ref, rtol=1e-15)
class TestNBinom:
    # 设置方法：初始化随机种子为1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试随机变量生成函数
    def test_rvs(self):
        # 生成形状为(2, 50)的负二项分布随机变量
        vals = stats.nbinom.rvs(10, 0.75, size=(2, 50))
        # 断言所有值都大于等于0
        assert_(np.all(vals >= 0))
        # 断言生成的数组形状为(2, 50)
        assert_(np.shape(vals) == (2, 50))
        # 断言生成的数组元素类型为整数
        assert_(vals.dtype.char in typecodes['AllInteger'])
        # 生成一个负二项分布的随机变量
        val = stats.nbinom.rvs(10, 0.75)
        # 断言该随机变量的类型为整数
        assert_(isinstance(val, int))
        # 使用分布对象生成三个随机变量
        val = stats.nbinom(10, 0.75).rvs(3)
        # 断言生成的对象类型为 numpy 数组
        assert_(isinstance(val, np.ndarray))
        # 断言生成的数组元素类型为整数
        assert_(val.dtype.char in typecodes['AllInteger'])

    # 测试概率质量函数
    def test_pmf(self):
        # 回归测试，检查负二项分布的概率质量函数与对数概率质量函数的指数和负二项分布的概率质量函数之间的近似度
        assert_allclose(np.exp(stats.nbinom.logpmf(700, 721, 0.52)),
                        stats.nbinom.pmf(700, 721, 0.52))
        # 对于参数为(0,1,1)的负二项分布的对数概率质量函数，不应返回 NaN（回归测试）
        val = scipy.stats.nbinom.logpmf(0, 1, 1)
        assert_equal(val, 0)

    # 测试对数累积分布函数，修复 gh16159
    def test_logcdf_gh16159(self):
        # 检查是否解决了 gh16159 问题
        vals = stats.nbinom.logcdf([0, 5, 0, 5], n=4.8, p=0.45)
        ref = np.log(stats.nbinom.cdf([0, 5, 0, 5], n=4.8, p=0.45))
        assert_allclose(vals, ref)


class TestGenInvGauss:
    # 设置方法：初始化随机种子为1234
    def setup_method(self):
        np.random.seed(1234)

    # 标记为慢速测试：带有模式移位的随机变量生成
    @pytest.mark.slow
    def test_rvs_with_mode_shift(self):
        # ratio_unif 方法，带有模式移位
        gig = stats.geninvgauss(2.3, 1.5)
        # 使用 Kolmogorov-Smirnov 测试检查生成的随机变量与累积分布函数之间的偏差
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        # 断言 p 值大于0.05
        assert_equal(p > 0.05, True)

    # 标记为慢速测试：不带模式移位的随机变量生成
    @pytest.mark.slow
    def test_rvs_without_mode_shift(self):
        # ratio_unif 方法，不带模式移位
        gig = stats.geninvgauss(0.9, 0.75)
        # 使用 Kolmogorov-Smirnov 测试检查生成的随机变量与累积分布函数之间的偏差
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        # 断言 p 值大于0.05
        assert_equal(p > 0.05, True)

    # 标记为慢速测试：使用新的生成方法
    @pytest.mark.slow
    def test_rvs_new_method(self):
        # Hoermann / Leydold 的新生成算法
        gig = stats.geninvgauss(0.1, 0.2)
        # 使用 Kolmogorov-Smirnov 测试检查生成的随机变量与累积分布函数之间的偏差
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        # 断言 p 值大于0.05
        assert_equal(p > 0.05, True)

    # 测试当 p = 0 时的随机变量生成
    @pytest.mark.slow
    def test_rvs_p_zero(self):
        def my_ks_check(p, b):
            # 当 p = 0 时的生成算法
            gig = stats.geninvgauss(p, b)
            # 生成随机变量并进行 Kolmogorov-Smirnov 测试
            rvs = gig.rvs(size=1500, random_state=1234)
            return stats.kstest(rvs, gig.cdf)[1] > 0.05

        # 当 p = 0 时的边界情况测试
        assert_equal(my_ks_check(0, 0.2), True)  # 新算法
        assert_equal(my_ks_check(0, 0.9), True)  # ratio_unif 方法，不带移位
        assert_equal(my_ks_check(0, 1.5), True)  # ratio_unif 方法，带移位

    # 测试当 p 为负数时的随机变量生成
    def test_rvs_negative_p(self):
        # 如果 p 为负数，返回其倒数
        assert_equal(
                stats.geninvgauss(-1.5, 2).rvs(size=10, random_state=1234),
                1 / stats.geninvgauss(1.5, 2).rvs(size=10, random_state=1234))
    def test_invgauss(self):
        # 测试逆高斯分布的特殊情况
        # 生成逆高斯分布样本
        ig = stats.geninvgauss.rvs(size=1500, p=-0.5, b=1, random_state=1234)
        # 使用 Kolmogorov-Smirnov 测试逆高斯分布的拟合情况
        assert_equal(stats.kstest(ig, 'invgauss', args=[1])[1] > 0.15, True)
        
        # 测试概率密度函数和累积分布函数
        mu, x = 100, np.linspace(0.01, 1, 10)
        # 计算逆高斯分布的概率密度函数
        pdf_ig = stats.geninvgauss.pdf(x, p=-0.5, b=1 / mu, scale=mu)
        # 检验与标准逆高斯分布的概率密度函数的接近程度
        assert_allclose(pdf_ig, stats.invgauss(mu).pdf(x))
        
        # 计算逆高斯分布的累积分布函数
        cdf_ig = stats.geninvgauss.cdf(x, p=-0.5, b=1 / mu, scale=mu)
        # 检验与标准逆高斯分布的累积分布函数的接近程度
        assert_allclose(cdf_ig, stats.invgauss(mu).cdf(x))

    def test_pdf_R(self):
        # 使用 R 包 GIGrvg 进行测试
        # 对比 R 中的 GIGrvg::dgig(x, 0.5, 1, 1)
        vals_R = np.array([2.081176820e-21, 4.488660034e-01, 3.747774338e-01,
                           2.693297528e-01, 1.905637275e-01, 1.351476913e-01,
                           9.636538981e-02, 6.909040154e-02, 4.978006801e-02,
                           3.602084467e-02])
        x = np.linspace(0.01, 5, 10)
        # 检验逆高斯分布的概率密度函数是否与 R 中的值接近
        assert_allclose(vals_R, stats.geninvgauss.pdf(x, 0.5, 1))

    def test_pdf_zero(self):
        # 对于 x = 0，概率密度函数为 0，需要特殊处理避免在计算中出现 1/x
        assert_equal(stats.geninvgauss.pdf(0, 0.5, 0.5), 0)
        
        # 如果 x 很大且 p 适中，确保概率密度函数不因 x**(p-1) 而溢出；
        # exp(-b*x) 强制概率密度函数趋向于零
        assert_equal(stats.geninvgauss.pdf(2e6, 50, 2), 0)
class TestGenHyperbolic:
    # 设置方法：初始化随机种子，以便测试可重复
    def setup_method(self):
        np.random.seed(1234)

    # 测试概率密度函数与 R 包 GeneralizedHyperbolic 的对比
    def test_pdf_r(self):
        # test against R package GeneralizedHyperbolic
        # x <- seq(-10, 10, length.out = 10)
        # GeneralizedHyperbolic::dghyp(
        #    x = x, lambda = 2, alpha = 2, beta = 1, delta = 1.5, mu = 0.5
        # )
        # 预先计算的 R 包返回值
        vals_R = np.array([
            2.94895678275316e-13, 1.75746848647696e-10, 9.48149804073045e-08,
            4.17862521692026e-05, 0.0103947630463822, 0.240864958986839,
            0.162833527161649, 0.0374609592899472, 0.00634894847327781,
            0.000941920705790324
            ])

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        # 使用 scipy.stats.genhyperbolic 创建广义双曲分布对象 gh
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        # 生成等间隔的测试数据
        x = np.linspace(-10, 10, 10)

        # 断言广义双曲分布的概率密度函数与预期值 vals_R 接近
        assert_allclose(gh.pdf(x), vals_R, atol=0, rtol=1e-13)

    # 测试累积分布函数与 R 包 GeneralizedHyperbolic 的对比
    def test_cdf_r(self):
        # test against R package GeneralizedHyperbolic
        # q <- seq(-10, 10, length.out = 10)
        # GeneralizedHyperbolic::pghyp(
        #   q = q, lambda = 2, alpha = 2, beta = 1, delta = 1.5, mu = 0.5
        # )
        # 预先计算的 R 包返回值
        vals_R = np.array([
            1.01881590921421e-13, 6.13697274983578e-11, 3.37504977637992e-08,
            1.55258698166181e-05, 0.00447005453832497, 0.228935323956347,
            0.755759458895243, 0.953061062884484, 0.992598013917513,
            0.998942646586662
            ])

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        # 使用 scipy.stats.genhyperbolic 创建广义双曲分布对象 gh
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        # 生成等间隔的测试数据
        x = np.linspace(-10, 10, 10)

        # 断言广义双曲分布的累积分布函数与预期值 vals_R 接近
        assert_allclose(gh.cdf(x), vals_R, atol=0, rtol=1e-6)

    # 使用 mpmath 实现的概率密度函数计算，并通过 mp.quad 进行积分验证
    # 引用值是通过设置 mp.dps=250 和 mp.dps=400 来确保完整的 64 位精度计算得到的。
    @pytest.mark.parametrize(
        'x, p, a, b, loc, scale, ref',
        [(-15, 2, 3, 1.5, 0.5, 1.5, 4.770036428808252e-20),
         (-15, 10, 1.5, 0.25, 1, 5, 0.03282964575089294),
         (-15, 10, 1.5, 1.375, 0, 1, 3.3711159600215594e-23),
         (-15, 0.125, 1.5, 1.49995, 0, 1, 4.729401428898605e-23),
         (-1, 0.125, 1.5, 1.49995, 0, 1, 0.0003565725914786859),
         (5, -0.125, 1.5, 1.49995, 0, 1, 0.2600651974023352),
         (5, -0.125, 1000, 999, 0, 1, 5.923270556517253e-28),
         (20, -0.125, 1000, 999, 0, 1, 0.23452293711665634),
         (40, -0.125, 1000, 999, 0, 1, 0.9999648749561968),
         (60, -0.125, 1000, 999, 0, 1, 0.9999999999975475)]
    )
    # 使用 mpmath 计算的累积分布函数，与预期值 ref 进行比较
    def test_cdf_mpmath(self, x, p, a, b, loc, scale, ref):
        cdf = stats.genhyperbolic.cdf(x, p, a, b, loc=loc, scale=scale)
        assert_allclose(cdf, ref, rtol=5e-12)
    # 使用 pytest 的 @parametrize 装饰器，为 test_sf_mpmath 方法定义多组参数化测试数据
    @pytest.mark.parametrize(
        'x, p, a, b, loc, scale, ref',
        [(0, 1e-6, 12, -1, 0, 1, 0.38520358671350524),  # 参数化测试数据集合
         (-1, 3, 2.5, 2.375, 1, 3, 0.9999901774267577),
         (-20, 3, 2.5, 2.375, 1, 3, 1.0),
         (25, 2, 3, 1.5, 0.5, 1.5, 8.593419916523976e-10),
         (300, 10, 1.5, 0.25, 1, 5, 6.137415609872158e-24),
         (60, -0.125, 1000, 999, 0, 1, 2.4524915075944173e-12),
         (75, -0.125, 1000, 999, 0, 1, 2.9435194886214633e-18)]
    )
    # 定义测试方法 test_sf_mpmath，计算并断言广义双曲分布的生存函数值是否接近参考值
    def test_sf_mpmath(self, x, p, a, b, loc, scale, ref):
        # 计算广义双曲分布的生存函数值
        sf = stats.genhyperbolic.sf(x, p, a, b, loc=loc, scale=scale)
        # 使用 assert_allclose 断言计算值与参考值的接近程度
        assert_allclose(sf, ref, rtol=5e-12)

    # 定义测试方法 test_moments_r，验证与 R 软件包 GeneralizedHyperbolic 的矩匹配
    def test_moments_r(self):
        # R 包 GeneralizedHyperbolic 的原始矩值
        vals_R = [2.36848366948115, 8.4739346779246,
                  37.8870502710066, 205.76608511485]

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        # 计算与 R 包对比的广义双曲分布矩值
        vals_us = [
            stats.genhyperbolic(*args, loc=mu, scale=delta).moment(i)
            for i in range(1, 5)
            ]

        # 使用 assert_allclose 断言计算的矩值与 R 包的矩值接近
        assert_allclose(vals_us, vals_R, atol=0, rtol=1e-13)

    # 定义测试方法 test_rvs，使用 Kolmogorov-Smirnov 检验分析与实验 CDF 的一致性
    def test_rvs(self):
        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        # 创建广义双曲分布对象
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        # 使用 Kolmogorov-Smirnov 检验
        _, p = stats.kstest(gh.rvs(size=1500, random_state=1234), gh.cdf)

        # 使用 assert_equal 断言检验结果是否大于显著性水平
        assert_equal(p > 0.05, True)

    # 定义测试方法 test_pdf_t，使用 T-Student 分布进行验证
    def test_pdf_t(self):
        # 创建一组自由度参数
        df = np.linspace(1, 30, 10)

        # 确定 alpha 和 beta 参数，以及 mu 和 delta 参数
        alpha, beta = np.float_power(df, 2)*np.finfo(np.float32).eps, 0
        mu, delta = 0, np.sqrt(df)
        args = (-df/2, alpha, beta)

        # 创建广义双曲分布对象
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        # 使用 assert_allclose 断言广义双曲分布的概率密度函数与 T-Student 分布的概率密度函数的接近程度
        assert_allclose(
            gh.pdf(x), stats.t.pdf(x, df),
            atol=0, rtol=1e-6
            )
    def test_pdf_cauchy(self):
        # Test Against Cauchy distribution
        
        # 设置分布参数 lmbda=-0.5, alpha=最小浮点数，beta=0
        lmbda, alpha, beta = -0.5, np.finfo(np.float32).eps, 0
        mu, delta = 0, 1
        args = (lmbda, alpha, beta)

        # 使用给定参数创建广义双曲线分布对象 gh
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        
        # 生成等间距的样本点 x，用于计算概率密度函数
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        # 断言广义双曲线分布的概率密度函数与 Cauchy 分布的概率密度函数在指定精度下的近似一致性
        assert_allclose(
            gh.pdf(x), stats.cauchy.pdf(x),
            atol=0, rtol=1e-6
            )

    def test_pdf_laplace(self):
        # Test Against Laplace with location param [-10, 10]
        
        # 设置位置参数 loc 为 -10 到 10 的等间距数列
        loc = np.linspace(-10, 10, 10)

        # 设置分布参数 delta=最小浮点数
        delta = np.finfo(np.float32).eps

        lmbda, alpha, beta = 1, 1, 0
        args = (lmbda, alpha*delta, beta*delta)

        # 使用给定参数创建广义双曲线分布对象 gh
        gh = stats.genhyperbolic(*args, loc=loc, scale=delta)
        
        # 生成等间距的样本点 x，用于计算概率密度函数
        x = np.linspace(-20, 20, 50)[:, np.newaxis]

        # 断言广义双曲线分布的概率密度函数与 Laplace 分布的概率密度函数在指定精度下的近似一致性
        assert_allclose(
            gh.pdf(x), stats.laplace.pdf(x, loc=loc, scale=1),
            atol=0, rtol=1e-11
            )

    def test_pdf_norminvgauss(self):
        # Test Against NIG with varying alpha/beta/delta/mu
        
        # 设置参数 alpha, beta, delta, mu 的范围
        alpha, beta, delta, mu = (
                np.linspace(1, 20, 10),
                np.linspace(0, 19, 10)*np.float_power(-1, range(10)),
                np.linspace(1, 1, 10),
                np.linspace(-100, 100, 10)
                )

        lmbda = - 0.5
        args = (lmbda, alpha * delta, beta * delta)

        # 使用给定参数创建广义双曲线分布对象 gh
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        
        # 生成等间距的样本点 x，用于计算概率密度函数
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        # 断言广义双曲线分布的概率密度函数与 Norminvgauss 分布的概率密度函数在指定精度下的近似一致性
        assert_allclose(
            gh.pdf(x), stats.norminvgauss.pdf(
                x, a=alpha, b=beta, loc=mu, scale=delta),
            atol=0, rtol=1e-13
            )
class TestHypSecant:

    # Reference values were computed with the mpmath expression
    #     float((2/mp.pi)*mp.atan(mp.exp(-x)))
    # and mp.dps = 50.
    @pytest.mark.parametrize('x, reference',
                             [(30, 5.957247804324683e-14),
                              (50, 1.2278802891647964e-22)])
    def test_sf(self, x, reference):
        # Calculate the survival function (sf) of the hyperbolic secant distribution
        sf = stats.hypsecant.sf(x)
        # Assert that the calculated sf is close to the reference value with a relative tolerance
        assert_allclose(sf, reference, rtol=5e-15)

    # Reference values were computed with the mpmath expression
    #     float(-mp.log(mp.tan((mp.pi/2)*p)))
    # and mp.dps = 50.
    @pytest.mark.parametrize('p, reference',
                             [(1e-6, 13.363927852673998),
                              (1e-12, 27.179438410639094)])
    def test_isf(self, p, reference):
        # Calculate the inverse survival function (isf) of the hyperbolic secant distribution
        x = stats.hypsecant.isf(p)
        # Assert that the calculated isf is close to the reference value with a relative tolerance
        assert_allclose(x, reference, rtol=5e-15)


class TestNormInvGauss:
    def setup_method(self):
        # Set a fixed seed for numpy random number generator
        np.random.seed(1234)

    def test_cdf_R(self):
        # Compare cumulative distribution function (cdf) values against R
        # These values are obtained from R package GeneralizedHyperbolic for NormInvGauss distribution
        r_cdf = np.array([8.034920282e-07, 2.512671945e-05, 3.186661051e-01,
                          9.988650664e-01, 9.999848769e-01])
        x_test = np.array([-7, -5, 0, 8, 15])
        # Calculate cdf values using the NormInvGauss distribution implementation
        vals_cdf = stats.norminvgauss.cdf(x_test, a=1, b=0.5)
        # Assert that calculated cdf values are close to the R values with an absolute tolerance
        assert_allclose(vals_cdf, r_cdf, atol=1e-9)

    def test_pdf_R(self):
        # Compare probability density function (pdf) values against R
        # These values are obtained from R package GeneralizedHyperbolic for NormInvGauss distribution
        r_pdf = np.array([1.359600783e-06, 4.413878805e-05, 4.555014266e-01,
                          7.450485342e-04, 8.917889931e-06])
        x_test = np.array([-7, -5, 0, 8, 15])
        # Calculate pdf values using the NormInvGauss distribution implementation
        vals_pdf = stats.norminvgauss.pdf(x_test, a=1, b=0.5)
        # Assert that calculated pdf values are close to the R values with an absolute tolerance
        assert_allclose(vals_pdf, r_pdf, atol=1e-9)

    @pytest.mark.parametrize('x, a, b, sf, rtol',
                             [(-1, 1, 0, 0.8759652211005315, 1e-13),
                              (25, 1, 0, 1.1318690184042579e-13, 1e-4),
                              (1, 5, -1.5, 0.002066711134653577, 1e-12),
                              (10, 5, -1.5, 2.308435233930669e-29, 1e-9)])
    def test_sf_isf_mpmath(self, x, a, b, sf, rtol):
        # Reference data generated with `reference_distributions.NormInvGauss`,
        # e.g. `NormInvGauss(alpha=1, beta=0).sf(-1)` with mp.dps = 50
        # Calculate the survival function (sf) using the NormInvGauss distribution
        s = stats.norminvgauss.sf(x, a, b)
        # Assert that the calculated sf is close to the reference sf with a relative tolerance
        assert_allclose(s, sf, rtol=rtol)
        # Calculate the inverse survival function (isf) using the NormInvGauss distribution
        i = stats.norminvgauss.isf(sf, a, b)
        # Assert that the calculated isf is close to the original x with a relative tolerance
        assert_allclose(i, x, rtol=rtol)
    # 定义一个测试函数，用于测试 norminvgauss 分布的 sf, isf 方法的向量化行为
    def test_sf_isf_mpmath_vectorized(self):
        # 定义输入参数 x, a, b
        x = [-1, 25]
        a = [1, 1]
        b = 0
        # 预期的 sf 值，参考之前的测试结果
        sf = [0.8759652211005315, 1.1318690184042579e-13]
        # 计算使用 norminvgauss 分布对象的 sf 方法得到的值
        s = stats.norminvgauss.sf(x, a, b)
        # 断言 s 与预期 sf 值的接近程度，指定相对和绝对误差容忍度
        assert_allclose(s, sf, rtol=1e-13, atol=1e-16)
        # 使用 norminvgauss 分布对象的 isf 方法计算逆 sf 值
        i = stats.norminvgauss.isf(sf, a, b)
        # 断言逆 sf 值 i 与输入 x 的接近程度，指定相对误差容忍度
        # 注意：这里的容忍度较宽，因为不一定能完全回到原始输入值 x
        assert_allclose(i, x, rtol=1e-6)

    # 定义一个测试函数，验证 gh-13338 是否解决了 gh-8718 问题
    def test_gh8718(self):
        # 创建一个 norminvgauss 分布对象 dst，参数为 (1, 0)
        dst = stats.norminvgauss(1, 0)
        # 创建一个 numpy 数组 x，包含从 0 到 18（步长为 2）的值
        x = np.arange(0, 20, 2)
        # 使用 norminvgauss 对象的 sf 方法计算 x 数组中每个值的 sf 值
        sf = dst.sf(x)
        # 使用 norminvgauss 对象的 isf 方法计算 sf 值的逆值
        isf = dst.isf(sf)
        # 断言逆 sf 值 isf 与输入 x 的接近程度
        assert_allclose(isf, x)

    # 定义一个测试函数，验证 norminvgauss 的 stats 方法返回的统计量是否正确
    def test_stats(self):
        # 定义参数 a 和 b
        a, b = 1, 0.5
        # 计算 gamma 值
        gamma = np.sqrt(a**2 - b**2)
        # 计算返回的统计量 v_stats，包括均值、方差、偏度和峰度
        v_stats = (b / gamma, a**2 / gamma**3, 3.0 * b / (a * np.sqrt(gamma)),
                   3.0 * (1 + 4 * b**2 / a**2) / gamma)
        # 使用 norminvgauss 的 stats 方法计算相同的统计量
        # 并断言它们与预期的 v_stats 接近
        assert_equal(v_stats, stats.norminvgauss.stats(a, b, moments='mvsk'))

    # 定义一个测试函数，验证 norminvgauss 的 ppf 方法的正确性
    def test_ppf(self):
        # 定义参数 a 和 b
        a, b = 1, 0.5
        # 创建一个测试用的 x_test 数组，包含三个概率值
        x_test = np.array([0.001, 0.5, 0.999])
        # 使用 norminvgauss 的 ppf 方法计算 x_test 数组对应的百分位点值
        vals = stats.norminvgauss.ppf(x_test, a, b)
        # 使用 norminvgauss 的 cdf 方法计算 ppf 返回的值的累积分布函数值
        # 并断言它们与输入的 x_test 接近
        assert_allclose(x_test, stats.norminvgauss.cdf(vals, a, b))
class TestGeom:
    # 设置方法，在每个测试方法执行前种下随机数种子，以确保测试的可重复性
    def setup_method(self):
        np.random.seed(1234)

    # 测试几何分布的随机变量生成
    def test_rvs(self):
        # 生成形状为 (2, 50) 的几何分布随机变量，验证所有值都大于等于 0
        vals = stats.geom.rvs(0.75, size=(2, 50))
        assert_(np.all(vals >= 0))
        # 验证生成的随机变量数组形状为 (2, 50)
        assert_(np.shape(vals) == (2, 50))
        # 验证生成的随机变量的数据类型是整数
        assert_(vals.dtype.char in typecodes['AllInteger'])
        # 生成单个几何分布随机变量，验证其类型为整数
        val = stats.geom.rvs(0.75)
        assert_(isinstance(val, int))
        # 生成形状为 (3,) 的几何分布随机变量数组，验证其类型为 numpy 数组
        val = stats.geom(0.75).rvs(3)
        assert_(isinstance(val, np.ndarray))
        # 验证生成的随机变量数组的数据类型是整数
        assert_(val.dtype.char in typecodes['AllInteger'])

    # 测试修复了 gh-9313 的几何分布随机变量生成问题
    def test_rvs_9313(self):
        # 使用指定的随机数生成器种子创建随机数生成器
        rng = np.random.default_rng(649496242618848)
        # 以极小的概率参数生成几何分布随机变量，验证数据类型为 np.int64
        rvs = stats.geom.rvs(np.exp(-35), size=5, random_state=rng)
        assert rvs.dtype == np.int64
        # 验证所有生成的随机变量都大于 np.int32 的最大值
        assert np.all(rvs > np.iinfo(np.int32).max)

    # 测试几何分布的概率质量函数
    def test_pmf(self):
        # 计算几何分布在给定点的概率质量函数值，验证与预期值的近似相等
        vals = stats.geom.pmf([1, 2, 3], 0.5)
        assert_array_almost_equal(vals, [0.5, 0.25, 0.125])

    # 测试几何分布的对数概率质量函数
    def test_logpmf(self):
        # 验证修复了票号 1793 的几何分布对数概率质量函数的问题
        vals1 = np.log(stats.geom.pmf([1, 2, 3], 0.5))
        vals2 = stats.geom.logpmf([1, 2, 3], 0.5)
        assert_allclose(vals1, vals2, rtol=1e-15, atol=0)
        # 验证修复了 gh-4028 的几何分布对数概率质量函数的问题
        val = stats.geom.logpmf(1, 1)
        assert_equal(val, 0.0)

    # 测试几何分布的累积分布函数和生存函数
    def test_cdf_sf(self):
        # 计算几何分布在给定点的累积分布函数和生存函数值，验证与预期值的近似相等
        vals = stats.geom.cdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.sf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, expected)
        assert_array_almost_equal(vals_sf, 1 - expected)

    # 测试几何分布的对数累积分布函数和对数生存函数
    def test_logcdf_logsf(self):
        # 计算几何分布在给定点的对数累积分布函数和对数生存函数值，验证与预期值的近似相等
        vals = stats.geom.logcdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.logsf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, np.log(expected))
        assert_array_almost_equal(vals_sf, np.log1p(-expected))

    # 测试几何分布的百分点函数
    def test_ppf(self):
        # 计算几何分布在给定概率下的百分点函数值，验证与预期值的近似相等
        vals = stats.geom.ppf([0.5, 0.75, 0.875], 0.5)
        expected = array([1.0, 2.0, 3.0])
        assert_array_almost_equal(vals, expected)

    # 测试修复了低概率下几何分布熵计算问题的问题
    def test_entropy_gh18226(self):
        # 验证修复了 gh-18226 报告的几何分布熵计算的问题
        h = stats.geom(0.0146).entropy()
        assert_allclose(h, 5.219397961962308, rtol=1e-15)


class TestPlanck:
    # 设置方法，在每个测试方法执行前种下随机数种子，以确保测试的可重复性
    def setup_method(self):
        np.random.seed(1234)

    # 测试普朗克分布的生存函数
    def test_sf(self):
        # 计算普朗克分布在给定点的生存函数值，验证与预期值的近似相等
        vals = stats.planck.sf([1, 2, 3], 5.)
        expected = array([4.5399929762484854e-05,
                          3.0590232050182579e-07,
                          2.0611536224385579e-09])
        assert_array_almost_equal(vals, expected)
    # 定义一个测试方法，用于测试 stats.planck.logsf 函数的行为
    def test_logsf(self):
        # 调用 stats.planck.logsf 函数，计算给定参数的 logsf 值
        vals = stats.planck.logsf([1000., 2000., 3000.], 1000.)
        # 预期的结果数组，用于与计算结果进行比较
        expected = array([-1001000., -2001000., -3001000.])
        # 使用 assert_array_almost_equal 函数断言计算结果与预期结果的近似相等性
        assert_array_almost_equal(vals, expected)
class TestGennorm:
    def test_laplace(self):
        # 对 gennorm 的 PDF 进行测试，与 Laplace 分布（特例 beta=1）进行比较
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 1)
        pdf2 = stats.laplace.pdf(points)
        assert_almost_equal(pdf1, pdf2)

    def test_norm(self):
        # 对 gennorm 的 PDF 进行测试，与正态分布（特例 beta=2）进行比较
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 2)
        pdf2 = stats.norm.pdf(points, scale=2**-.5)
        assert_almost_equal(pdf1, pdf2)

    def test_rvs(self):
        np.random.seed(0)
        # 0 < beta < 1 的情况
        dist = stats.gennorm(0.5)
        rvs = dist.rvs(size=1000)
        assert stats.kstest(rvs, dist.cdf).pvalue > 0.1
        # beta = 1 的情况
        dist = stats.gennorm(1)
        rvs = dist.rvs(size=1000)
        rvs_laplace = stats.laplace.rvs(size=1000)
        assert stats.ks_2samp(rvs, rvs_laplace).pvalue > 0.1
        # beta = 2 的情况
        dist = stats.gennorm(2)
        rvs = dist.rvs(size=1000)
        rvs_norm = stats.norm.rvs(scale=1/2**0.5, size=1000)
        assert stats.ks_2samp(rvs, rvs_norm).pvalue > 0.1

    def test_rvs_broadcasting(self):
        np.random.seed(0)
        # 对 gennorm 进行广播的随机变量生成测试
        dist = stats.gennorm([[0.5, 1.], [2., 5.]])
        rvs = dist.rvs(size=[1000, 2, 2])
        assert stats.kstest(rvs[:, 0, 0], stats.gennorm(0.5).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 0, 1], stats.gennorm(1.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 0], stats.gennorm(2.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 1], stats.gennorm(5.0).cdf)[1] > 0.1


class TestGibrat:

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def gibrat_sf(x):
    #       return 1 - mp.ncdf(mp.log(x))
    #
    # E.g.
    #
    #   >>> float(gibrat_sf(1.5))
    #   0.3425678305148459
    #
    @pytest.mark.parametrize('x, sfx', [(1.5, 0.3425678305148459),
                                        (5000, 8.173334352522493e-18)])
    def test_sf_isf(self, x, sfx):
        # 对 gibrat 分布的 sf 和 isf 进行测试
        assert_allclose(stats.gibrat.sf(x), sfx, rtol=2e-14)
        assert_allclose(stats.gibrat.isf(sfx), x, rtol=2e-14)


class TestGompertz:

    def test_gompertz_accuracy(self):
        # 对 gompertz 分布的精度进行回归测试（gh-4031）
        p = stats.gompertz.ppf(stats.gompertz.cdf(1e-100, 1), 1)
        assert_allclose(p, 1e-100)

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def gompertz_sf(x, c):
    #       return mp.exp(-c*mp.expm1(x))
    #
    # E.g.
    #
    #   >>> float(gompertz_sf(1, 2.5))
    #   0.013626967146253437
    #
    @pytest.mark.parametrize('x, c, sfx', [(1, 2.5, 0.013626967146253437),
                                           (3, 2.5, 1.8973243273704087e-21),
                                           (0.05, 5, 0.7738668242570479),
                                           (2.25, 5, 3.707795833465481e-19)])
    def test_sf(self, x, c, sfx):
        # 对 gompertz 分布的 sf 进行参数化测试
        assert_allclose(stats.gompertz.sf(x, c), sfx, rtol=1e-12)
    # 定义一个测试方法，用于测试 Gompertz 分布的生存函数 sf 和逆生存函数 isf 是否正确
    def test_sf_isf(self, x, c, sfx):
        # 使用 assert_allclose 函数验证 stats.gompertz.sf(x, c) 的计算结果是否与 sfx 接近，相对误差容忍度为 1e-14
        assert_allclose(stats.gompertz.sf(x, c), sfx, rtol=1e-14)
        # 使用 assert_allclose 函数验证 stats.gompertz.isf(sfx, c) 的计算结果是否与 x 接近，相对误差容忍度为 1e-14
        assert_allclose(stats.gompertz.isf(sfx, c), x, rtol=1e-14)

    # 参考值是使用 mpmath 计算得到的
    # 导入 mpmath 库
    # from mpmath import mp
    # 设置 mpmath 的精度为 100
    # mp.dps = 100
    # 定义一个函数 gompertz_entropy，用于计算 Gompertz 分布的熵
    # def gompertz_entropy(c):
    #     将 c 转换为 mpmath 的浮点数类型 mp.mpf(c)
    #     c = mp.mpf(c)
    #     计算并返回 Gompertz 分布的熵，使用 mpmath 提供的数学函数
    #     return float(mp.one - mp.log(c) - mp.exp(c)*mp.e1(c))

    # 使用 pytest 的 parametrize 装饰器，对 test_entropy 方法进行参数化测试
    @pytest.mark.parametrize('c, ref', [(1e-4, 1.5762523017634573),
                                        (1, 0.4036526376768059),
                                        (1000, -5.908754280976161),
                                        (1e10, -22.025850930040455)])
    # 定义测试方法 test_entropy，用于测试 stats.gompertz.entropy(c) 的计算结果是否与给定的参考值 ref 接近
    def test_entropy(self, c, ref):
        # 使用 assert_allclose 函数验证 stats.gompertz.entropy(c) 的计算结果是否与 ref 接近，相对误差容忍度为 1e-14
        assert_allclose(stats.gompertz.entropy(c), ref, rtol=1e-14)
class TestFoldNorm:

    # reference values were computed with mpmath with 50 digits of precision
    # from mpmath import mp
    # mp.dps = 50
    # mp.mpf(0.5) * (mp.erf((x - c)/mp.sqrt(2)) + mp.erf((x + c)/mp.sqrt(2)))

    @pytest.mark.parametrize('x, c, ref', [(1e-4, 1e-8, 7.978845594730578e-05),
                                           (1e-4, 1e-4, 7.97884555483635e-05)])
    def test_cdf(self, x, c, ref):
        assert_allclose(stats.foldnorm.cdf(x, c), ref, rtol=1e-15)


class TestHalfNorm:

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_sf(x):
    #       return 2*(1 - mp.ncdf(x))
    #
    # E.g.
    #
    #   >>> float(halfnorm_sf(1))
    #   0.3173105078629141
    #
    @pytest.mark.parametrize('x, sfx', [(1, 0.3173105078629141),
                                        (10, 1.523970604832105e-23)])
    def test_sf_isf(self, x, sfx):
        assert_allclose(stats.halfnorm.sf(x), sfx, rtol=1e-14)
        assert_allclose(stats.halfnorm.isf(sfx), x, rtol=1e-14)

    #   reference values were computed via mpmath
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_cdf_mpmath(x):
    #       x = mp.mpf(x)
    #       return float(mp.erf(x/mp.sqrt(2.)))

    @pytest.mark.parametrize('x, ref', [(1e-40, 7.978845608028653e-41),
                                        (1e-18, 7.978845608028654e-19),
                                        (8, 0.9999999999999988)])
    def test_cdf(self, x, ref):
        assert_allclose(stats.halfnorm.cdf(x), ref, rtol=1e-15)

    @pytest.mark.parametrize("rvs_loc", [1e-5, 1e10])
    @pytest.mark.parametrize("rvs_scale", [1e-2, 100, 1e8])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale,
                                    fix_loc, fix_scale):

        rng = np.random.default_rng(6762668991392531563)
        data = stats.halfnorm.rvs(loc=rvs_loc, scale=rvs_scale, size=1000,
                                  random_state=rng)

        if fix_loc and fix_scale:
            error_msg = ("All parameters fixed. There is nothing to "
                         "optimize.")
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halflogistic.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return

        kwds = {}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # Numerical result may equal analytical result if the initial guess
        # computed from moment condition is already optimal.
        _assert_less_or_close_loglike(stats.halfnorm, data, **kwds,
                                      maybe_identical=True)


注释：

# 测试Folded Normal分布的累积分布函数（CDF）是否正确
@pytest.mark.parametrize('x, c, ref', [(1e-4, 1e-8, 7.978845594730578e-05),
                                       (1e-4, 1e-4, 7.97884555483635e-05)])
def test_cdf(self, x, c, ref):
    assert_allclose(stats.foldnorm.cdf(x, c), ref, rtol=1e-15)


class TestHalfNorm:

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_sf(x):
    #       return 2*(1 - mp.ncdf(x))
    #
    # E.g.
    #
    #   >>> float(halfnorm_sf(1))
    #   0.3173105078629141
    #
    # 测试Half Normal分布的生存函数（SF）和逆生存函数（ISF）是否正确
    @pytest.mark.parametrize('x, sfx', [(1, 0.3173105078629141),
                                        (10, 1.523970604832105e-23)])
    def test_sf_isf(self, x, sfx):
        assert_allclose(stats.halfnorm.sf(x), sfx, rtol=1e-14)
        assert_allclose(stats.halfnorm.isf(sfx), x, rtol=1e-14)

    #   reference values were computed via mpmath
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_cdf_mpmath(x):
    #       x = mp.mpf(x)
    #       return float(mp.erf(x/mp.sqrt(2.)))

    # 测试Half Normal分布的累积分布函数（CDF）是否正确
    @pytest.mark.parametrize('x, ref', [(1e-40, 7.978845608028653e-41),
                                        (1e-18, 7.978845608028654e-19),
                                        (8, 0.9999999999999988)])
    def test_cdf(self, x, ref):
        assert_allclose(stats.halfnorm.cdf(x), ref, rtol=1e-15)

    # 测试用最大似然估计（MLE）配合优化器的拟合结果是否正确
    @pytest.mark.parametrize("rvs_loc", [1e-5, 1e10])
    @pytest.mark.parametrize("rvs_scale", [1e-2, 100, 1e8])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale,
                                    fix_loc, fix_scale):

        rng = np.random.default_rng(6762668991392531563)
        # 生成Half Normal分布的随机样本数据
        data = stats.halfnorm.rvs(loc=rvs_loc, scale=rvs_scale, size=1000,
                                  random_state=rng)

        if fix_loc and fix_scale:
            error_msg = ("All parameters fixed. There is nothing to "
                         "optimize.")
            # 如果所有参数都被固定，测试应该引发RuntimeError
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halflogistic.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return

        kwds = {}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 如果从矩条件计算的初始猜测已经是最优的，数值结果可能等于分析结果。
        # 检查Half Normal分布的拟合结果的对数似然是否小于或接近预期值
        _assert_less_or_close_loglike(stats.halfnorm, data, **kwds,
                                      maybe_identical=True)
    def test_fit_error(self):
        # 使用 pytest 的上下文管理器 `raises` 来验证是否会抛出 FitDataError 异常
        with pytest.raises(FitDataError):
            # 调用 halfnorm.fit 方法，传入数据 [1, 2, 3] 并设置参数 floc=2
            # 预期这里会抛出 FitDataError 异常，因为 floc 参数大于数据中的最小值
            stats.halfnorm.fit([1, 2, 3], floc=2)
    @pytest.mark.parametrize("rvs_loc", [1e-5, 1e10])
    @pytest.mark.parametrize("rvs_scale", [1e-2, 100, 1e8])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale,
                                    fix_loc, fix_scale):
        # 使用指定的种子创建随机数生成器对象
        rng = np.random.default_rng(6762668991392531563)
        # 生成服从半正态分布的随机数据
        data = stats.halfnorm.rvs(loc=rvs_loc, scale=rvs_scale, size=1000,
                                  random_state=rng)

        # 如果同时固定位置参数和尺度参数，则抛出运行时错误
        if fix_loc and fix_scale:
            error_msg = ("All parameters fixed. There is nothing to "
                         "optimize.")
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halfcauchy.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return

        kwds = {}
        # 如果固定位置参数，则将其添加到参数字典中
        if fix_loc:
            kwds['floc'] = rvs_loc
        # 如果固定尺度参数，则将其添加到参数字典中
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 调用自定义的断言函数，验证半柯西分布的对数似然函数
        _assert_less_or_close_loglike(stats.halfcauchy, data, **kwds)

    def test_fit_error(self):
        # 当位置参数 `floc` 大于数据中的最小值时，应抛出适应数据错误
        with pytest.raises(FitDataError):
            stats.halfcauchy.fit([1, 2, 3], floc=2)
    # 定义一个测试函数，用于测试使用最大似然估计（MLE）配合优化器的情况
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale,
                                    fix_loc, fix_scale):
        # 使用特定种子创建随机数生成器对象
        rng = np.random.default_rng(6762668991392531563)
        # 生成服从半对数分布的随机数据，指定位置参数和尺度参数
        data = stats.halflogistic.rvs(loc=rvs_loc, scale=rvs_scale, size=1000,
                                      random_state=rng)

        # 初始化空的关键字参数字典
        kwds = {}
        # 如果固定了位置参数和尺度参数，则抛出运行时错误，不进行优化
        if fix_loc and fix_scale:
            error_msg = ("All parameters fixed. There is nothing to "
                         "optimize.")
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halflogistic.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return

        # 如果仅固定位置参数，则设置关键字参数字典中的位置参数
        if fix_loc:
            kwds['floc'] = rvs_loc
        # 如果仅固定尺度参数，则设置关键字参数字典中的尺度参数
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 断言：如果初始猜测从矩条件计算得到的结果已经是最优的，数值结果可能等于分析结果
        _assert_less_or_close_loglike(stats.halflogistic, data, **kwds,
                                      maybe_identical=True)

    # 定义一个测试函数，用于测试在给定错误位置参数时的最大似然估计
    def test_fit_bad_floc(self):
        # 定义错误消息的正则表达式，用于匹配错误信息
        msg = r" Maximum likelihood estimation with 'halflogistic' requires"
        # 断言：调用最大似然估计时，传入错误的位置参数会引发特定的异常
        with assert_raises(FitDataError, match=msg):
            stats.halflogistic.fit([0, 2, 4], floc=1)
class TestHalfgennorm:
    def test_expon(self):
        # 对指数分布进行测试（beta=1时的特殊情况）
        points = [1, 2, 3]
        # 使用 stats.halfgennorm.pdf 计算概率密度函数
        pdf1 = stats.halfgennorm.pdf(points, 1)
        # 使用 stats.expon.pdf 计算指数分布的概率密度函数
        pdf2 = stats.expon.pdf(points)
        # 断言两个概率密度函数的近似相等
        assert_almost_equal(pdf1, pdf2)

    def test_halfnorm(self):
        # 对半正态分布进行测试（beta=2时的特殊情况）
        points = [1, 2, 3]
        # 使用 stats.halfgennorm.pdf 计算概率密度函数
        pdf1 = stats.halfgennorm.pdf(points, 2)
        # 使用 stats.halfnorm.pdf 计算半正态分布的概率密度函数
        pdf2 = stats.halfnorm.pdf(points, scale=2**-.5)
        # 断言两个概率密度函数的近似相等
        assert_almost_equal(pdf1, pdf2)

    def test_gennorm(self):
        # 对广义正态分布进行测试
        points = [1, 2, 3]
        # 使用 stats.halfgennorm.pdf 计算概率密度函数
        pdf1 = stats.halfgennorm.pdf(points, .497324)
        # 使用 stats.gennorm.pdf 计算广义正态分布的概率密度函数
        pdf2 = stats.gennorm.pdf(points, .497324)
        # 断言两个概率密度函数的近似相等
        assert_almost_equal(pdf1, 2*pdf2)


class TestLaplaceasymmetric:
    def test_laplace(self):
        # 对拉普拉斯分布进行测试（kappa=1时的特殊情况）
        points = np.array([1, 2, 3])
        # 使用 stats.laplace_asymmetric.pdf 计算概率密度函数
        pdf1 = stats.laplace_asymmetric.pdf(points, 1)
        # 使用 stats.laplace.pdf 计算拉普拉斯分布的概率密度函数
        pdf2 = stats.laplace.pdf(points)
        # 断言两个概率密度函数的全部近似相等
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_pdf(self):
        # 对非对称拉普拉斯分布进行测试
        points = np.array([1, 2, 3])
        kappa = 2
        kapinv = 1/kappa
        # 使用 stats.laplace_asymmetric.pdf 计算概率密度函数
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        # 使用 stats.laplace_asymmetric.pdf 计算概率密度函数
        pdf2 = stats.laplace_asymmetric.pdf(points*(kappa**2), kapinv)
        # 断言两个概率密度函数的全部近似相等
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_log_10_16(self):
        # 对非对称拉普拉斯分布进行测试
        points = np.array([-np.log(16), np.log(10)])
        kappa = 2
        # 使用 stats.laplace_asymmetric.pdf 计算概率密度函数
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        # 使用 stats.laplace_asymmetric.cdf 计算累积分布函数
        cdf1 = stats.laplace_asymmetric.cdf(points, kappa)
        # 使用 stats.laplace_asymmetric.sf 计算生存函数
        sf1 = stats.laplace_asymmetric.sf(points, kappa)
        pdf2 = np.array([1/10, 1/250])
        cdf2 = np.array([1/5, 1 - 1/500])
        sf2 = np.array([4/5, 1/500])
        # 使用 stats.laplace_asymmetric.ppf 计算分位点函数
        ppf1 = stats.laplace_asymmetric.ppf(cdf2, kappa)
        ppf2 = points
        # 使用 stats.laplace_asymmetric.isf 计算逆生存函数
        isf1 = stats.laplace_asymmetric.isf(sf2, kappa)
        isf2 = points
        # 断言各个数值数组的全部近似相等
        assert_allclose(np.concatenate((pdf1, cdf1, sf1, ppf1, isf1)),
                        np.concatenate((pdf2, cdf2, sf2, ppf2, isf2)))


class TestTruncnorm:
    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize("a, b, ref",
                             [(0, 100, 0.7257913526447274),
                             (0.6, 0.7, -2.3027610681852573),
                             (1e-06, 2e-06, -13.815510557964274)])
    def test_entropy(self, a, b, ref):
        # 对熵函数进行测试，验证计算结果与参考值的接近程度
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-10)

    @pytest.mark.parametrize("a, b, ref",
                             [(1e-11, 10000000000.0, 0.725791352640738),
                             (1e-100, 1e+100, 0.7257913526447274),
                             (-1e-100, 1e+100, 0.7257913526447274),
                             (-1e+100, 1e+100, 1.4189385332046727)])
    def test_extreme_entropy(self, a, b, ref):
        # 对极端情况下的熵函数进行测试，验证计算结果与参考值的接近程度
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-14)

    def test_ppf_ticket1131(self):
        # 测试在特定参数下的百分点函数（ppf）的计算结果
        vals = stats.truncnorm.ppf([-0.5, 0, 1e-4, 0.5, 1-1e-4, 1, 2], -1., 1.,
                                   loc=[3]*7, scale=2)
        expected = np.array([np.nan, 1, 1.00056419, 3, 4.99943581, 5, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_isf_ticket1131(self):
        # 测试在特定参数下的逆百分点函数（isf）的计算结果
        vals = stats.truncnorm.isf([-0.5, 0, 1e-4, 0.5, 1-1e-4, 1, 2], -1., 1.,
                                   loc=[3]*7, scale=2)
        expected = np.array([np.nan, 5, 4.99943581, 3, 1.00056419, 1, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_gh_2477_small_values(self):
        # 检查在小值情况下的特定问题，确保随机变量生成在指定范围内
        # 情况1：low=-11, high=-10
        low, high = -11, -10
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

        # 情况2：low=10, high=11
        low, high = 10, 11
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)
    def test_gh_2477_large_values(self):
        # 检查极端尾部情况是否会失败。
        low, high = 100, 101
        # 生成符合指定截断正态分布的随机变量
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        # 断言确保生成的随机变量在指定的范围内
        assert_(low <= x.min() <= x.max() <= high), str([low, high, x])

        # 再次检查额外的极端尾部情况
        low, high = 1000, 1001
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

        low, high = 10000, 10001
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

        low, high = -10001, -10000
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_9403_nontail_values(self):
        for low, high in [[3, 4], [-4, -3]]:
            # 定义一组特定边界和无尾情况下的测试值
            xvals = np.array([-np.inf, low, high, np.inf])
            xmid = (high+low)/2.0
            # 计算截断正态分布的累积分布函数、生存函数和概率密度函数
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            sfs = stats.truncnorm.sf(xvals, low, high)
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            expected_cdfs = np.array([0, 0, 1, 1])
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            expected_pdfs = np.array([0, 3.3619772, 0.1015229, 0])
            if low < 0:
                expected_pdfs = np.array([0, 0.1015229, 3.3619772, 0])
            # 断言检查计算得到的值与预期值的接近程度
            assert_almost_equal(cdfs, expected_cdfs)
            assert_almost_equal(sfs, expected_sfs)
            assert_almost_equal(pdfs, expected_pdfs)
            assert_almost_equal(np.log(expected_pdfs[1]/expected_pdfs[2]),
                                low + 0.5)
            # 计算截断正态分布的百分位点函数
            pvals = np.array([0, 0.5, 1.0])
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            expected_ppfs = np.array([low, np.sign(low)*3.1984741, high])
            assert_almost_equal(ppfs, expected_ppfs)

            if low < 0:
                # 如果边界低于零，验证生存函数和累积分布函数的值
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.8475544278436675)
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.1524455721563326)
            else:
                # 否则，验证累积分布函数和生存函数的值
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.8475544278436675)
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.1524455721563326)
            # 计算截断正态分布的概率密度函数
            pdf = stats.truncnorm.pdf(xmid, low, high)
            assert_almost_equal(np.log(pdf/expected_pdfs[2]), (xmid+0.25)/2)
    # 定义一个测试函数，用于验证 GitHub 问题 #9403 中截尾正态分布的边界情况
    def test_gh_9403_medium_tail_values(self):
        # 针对不同的截尾正态分布边界情况进行循环测试
        for low, high in [[39, 40], [-40, -39]]:
            # 创建一个包含四个特定值的 NumPy 数组，表示负无穷、截尾正态分布的下界、上界和正无穷
            xvals = np.array([-np.inf, low, high, np.inf])
            # 计算截尾正态分布的中点值
            xmid = (high+low)/2.0
            # 计算截尾正态分布的累积分布函数（CDF）
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            # 计算截尾正态分布的生存函数（SF）
            sfs = stats.truncnorm.sf(xvals, low, high)
            # 计算截尾正态分布的概率密度函数（PDF）
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            # 预期的累积分布函数结果
            expected_cdfs = np.array([0, 0, 1, 1])
            # 预期的生存函数结果
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            # 预期的概率密度函数结果
            expected_pdfs = np.array([0, 3.90256074e+01, 2.73349092e-16, 0])
            # 如果下界 low 小于 0，调整预期的概率密度函数结果
            if low < 0:
                expected_pdfs = np.array([0, 2.73349092e-16,
                                          3.90256074e+01, 0])
            # 断言实际得到的累积分布函数接近预期值
            assert_almost_equal(cdfs, expected_cdfs)
            # 断言实际得到的生存函数接近预期值
            assert_almost_equal(sfs, expected_sfs)
            # 断言实际得到的概率密度函数接近预期值
            assert_almost_equal(pdfs, expected_pdfs)
            # 断言对数值接近预期的比值
            assert_almost_equal(np.log(expected_pdfs[1]/expected_pdfs[2]),
                                low + 0.5)
            # 创建一个包含特定概率值的 NumPy 数组
            pvals = np.array([0, 0.5, 1.0])
            # 计算截尾正态分布的百分位点函数（PPF）
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            # 预期的百分位点函数结果
            expected_ppfs = np.array([low, np.sign(low)*39.01775731, high])
            # 断言实际得到的百分位点函数接近预期值
            assert_almost_equal(ppfs, expected_ppfs)
            # 计算使用百分位点计算得到的累积分布函数
            cdfs = stats.truncnorm.cdf(ppfs, low, high)
            # 断言实际得到的累积分布函数接近给定的概率值
            assert_almost_equal(cdfs, pvals)

            # 根据下界 low 的情况进行条件断言
            if low < 0:
                # 断言截尾正态分布的生存函数接近预期值
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.9999999970389126)
                # 断言截尾正态分布的累积分布函数接近预期值
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    2.961048103554866e-09)
            else:
                # 断言截尾正态分布的累积分布函数接近预期值
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.9999999970389126)
                # 断言截尾正态分布的生存函数接近预期值
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    2.961048103554866e-09)
            # 计算截尾正态分布的概率密度函数
            pdf = stats.truncnorm.pdf(xmid, low, high)
            # 断言实际得到的对数值接近预期的比值
            assert_almost_equal(np.log(pdf/expected_pdfs[2]), (xmid+0.25)/2)

            # 创建一个线性间隔的 NumPy 数组，用于测试截尾正态分布的对称性质
            xvals = np.linspace(low, high, 11)
            # 创建对称的 xvals2 数组
            xvals2 = -xvals[::-1]
            # 断言使用不同方式计算得到的累积分布函数接近
            assert_almost_equal(stats.truncnorm.cdf(xvals, low, high),
                                stats.truncnorm.sf(xvals2, -high, -low)[::-1])
            # 断言使用不同方式计算得到的生存函数接近
            assert_almost_equal(stats.truncnorm.sf(xvals, low, high),
                                stats.truncnorm.cdf(xvals2, -high, -low)[::-1])
            # 断言使用不同方式计算得到的概率密度函数接近
            assert_almost_equal(stats.truncnorm.pdf(xvals, low, high),
                                stats.truncnorm.pdf(xvals2, -high, -low)[::-1])
    # 定义测试函数，用于检查在 gh-14753 和 gh-155110 中报告的精度问题
    def test_cdf_tail_15110_14753(self):
        # 使用 Wolfram Alpha 计算的基准值，例如：
        # (CDF[NormalDistribution[0,1],83/10]-CDF[NormalDistribution[0,1],8])/
        #     (1 - CDF[NormalDistribution[0,1],8])
        assert_allclose(stats.truncnorm(13., 15.).cdf(14.),
                        0.9999987259565643)
        assert_allclose(stats.truncnorm(8, np.inf).cdf(8.3),
                        0.9163220907327540)

    # 用于测试 truncnorm stats() 方法的数据
    # 每行数据包括：
    #   a, b, 均值, 方差, 偏度, 峰度。使用以下链接生成：
    # https://gist.github.com/WarrenWeckesser/636b537ee889679227d53543d333a720
    _truncnorm_stats_data = [
        [-30, 30,
         0.0, 1.0, 0.0, 0.0],
        [-10, 10,
         0.0, 1.0, 0.0, -1.4927521335810455e-19],
        [-3, 3,
         0.0, 0.9733369246625415, 0.0, -0.17111443639774404],
        [-2, 2,
         0.0, 0.7737413035499232, 0.0, -0.6344632828703505],
        [0, np.inf,
         0.7978845608028654,
         0.3633802276324187,
         0.995271746431156,
         0.8691773036059741],
        [-np.inf, 0,
         -0.7978845608028654,
         0.3633802276324187,
         -0.995271746431156,
         0.8691773036059741],
        [-1, 3,
         0.282786110727154,
         0.6161417353578293,
         0.5393018494027877,
         -0.20582065135274694],
        [-3, 1,
         -0.282786110727154,
         0.6161417353578293,
         -0.5393018494027877,
         -0.20582065135274694],
        [-10, -9,
         -9.108456288012409,
         0.011448805821636248,
         -1.8985607290949496,
         5.0733461105025075],
    ]
    _truncnorm_stats_data = np.array(_truncnorm_stats_data)

    # 使用参数化测试来测试 moments() 方法
    @pytest.mark.parametrize("case", _truncnorm_stats_data)
    def test_moments(self, case):
        a, b, m0, v0, s0, k0 = case
        # 计算给定参数 a, b 的均值、方差、偏度和峰度
        m, v, s, k = stats.truncnorm.stats(a, b, moments='mvsk')
        # 检查计算结果是否与预期值相近，允许误差为 1e-17
        assert_allclose([m, v, s, k], [m0, v0, s0, k0], atol=1e-17)

    # 测试 truncnorm stats() 方法的特定情况
    def test_9902_moments(self):
        # 计算区间 [0, np.inf) 的均值和方差
        m, v = stats.truncnorm.stats(0, np.inf, moments='mv')
        # 检查均值是否接近预期值 0.79788456
        assert_almost_equal(m, 0.79788456)
        # 检查方差是否接近预期值 0.36338023
        assert_almost_equal(v, 0.36338023)

    # 检查原始示例
    def test_gh_1489_trac_962_rvs(self):
        # 检查原始示例
        low, high = 10, 15
        # 生成位于区间 [low, high) 内的随机样本，大小为 10
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        # 断言生成的随机样本在指定区间内
        assert_(low < x.min() < x.max() < high)

    # 由于研究 gh-11299 而来的测试
    def test_gh_11299_rvs(self):
        # 同时测试多个形状参数
        low = [-10, 10, -np.inf, -5, -np.inf, -np.inf, -45, -45, 40, -10, 40]
        high = [-5, 11, 5, np.inf, 40, -40, 40, -40, 45, np.inf, np.inf]
        # 生成多维随机样本，其形状由 low 的长度决定，共 5 组
        x = stats.truncnorm.rvs(low, high, size=(5, len(low)))
        # 检查生成的随机样本的形状是否符合预期
        assert np.shape(x) == (5, len(low))
        # 检查生成的随机样本每列的最小值是否大于等于对应的 low 值
        assert_(np.all(low <= x.min(axis=0)))
        # 检查生成的随机样本每列的最大值是否小于等于对应的 high 值
        assert_(np.all(x.max(axis=0) <= high))
    def test_rvs_Generator(self):
        # 检查 rvs 方法是否可以使用 Generator
        # 如果 np.random 模块有 default_rng 属性，则进行测试
        if hasattr(np.random, "default_rng"):
            # 使用 np.random.default_rng() 作为随机数生成器
            stats.truncnorm.rvs(-10, -5, size=5,
                                random_state=np.random.default_rng())

    def test_logcdf_gh17064(self):
        # 回归测试 gh-17064 - 避免 logcdf 大约为 0 的舍入误差
        # 定义输入数组 a, b 和 x
        a = np.array([-np.inf, -np.inf, -8, -np.inf, 10])
        b = np.array([np.inf, np.inf, 8, 10, np.inf])
        x = np.array([10, 7.5, 7.5, 9, 20])
        # 期望的结果数组
        expected = [-7.619853024160525e-24, -3.190891672910947e-14,
                    -3.128682067168231e-14, -1.1285122074235991e-19,
                    -3.61374964828753e-66]
        # 检查 stats.truncnorm.logcdf(x) 是否与期望结果相近
        assert_allclose(stats.truncnorm(a, b).logcdf(x), expected)
        # 检查 stats.truncnorm(-b, -a).logsf(-x) 是否与期望结果相近
        assert_allclose(stats.truncnorm(-b, -a).logsf(-x), expected)

    def test_moments_gh18634(self):
        # gh-18634 报告称，5阶及更高阶的矩计算存在问题；进行验证解决情况
        # 计算 stats.truncnorm(-2, 3).moment(5)
        res = stats.truncnorm(-2, 3).moment(5)
        # 从 Mathematica 得到的参考值
        ref = 1.645309620208361
        # 检查计算结果是否与参考值相近
        assert_allclose(res, ref)
# 定义一个测试类 TestGenLogistic，用于测试 genlogistic 分布的函数
class TestGenLogistic:

    # 使用 pytest.mark.parametrize 装饰器，为 test_logpdf 方法提供参数化测试数据
    # 每组参数包括输入值 x 和期望输出值 expected
    @pytest.mark.parametrize('x, expected', [(-1000, -1499.5945348918917),
                                             (-125, -187.09453489189184),
                                             (0, -1.3274028432916989),
                                             (100, -99.59453489189184),
                                             (1000, -999.5945348918918)])
    # 定义测试方法 test_logpdf，测试 genlogistic 分布的 logpdf 函数
    def test_logpdf(self, x, expected):
        # 设置参数 c 的值
        c = 1.5
        # 调用 genlogistic 模块的 logpdf 函数，计算给定 x 和 c 的结果
        logp = stats.genlogistic.logpdf(x, c)
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(logp, expected, rtol=1e-13)

    # 使用 pytest.mark.parametrize 装饰器，为 test_entropy 方法提供参数化测试数据
    # 每组参数包括输入值 c 和期望输出值 ref
    @pytest.mark.parametrize('c, ref', [(1e-100, 231.25850929940458),
                                        (1e-4, 10.21050485336338),
                                        (1e8, 1.577215669901533),
                                        (1e100, 1.5772156649015328)])
    # 定义测试方法 test_entropy，测试 genlogistic 分布的 entropy 函数
    def test_entropy(self, c, ref):
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(stats.genlogistic.entropy(c), ref, rtol=5e-15)

    # 使用 pytest.mark.parametrize 装饰器，为 test_sf 方法提供参数化测试数据
    # 每组参数包括输入值 x, c 和期望输出值 ref
    @pytest.mark.parametrize('x, c, ref', [(200, 10, 1.3838965267367375e-86),
                                           (500, 20, 1.424915281348257e-216)])
    # 定义测试方法 test_sf，测试 genlogistic 分布的 sf 函数
    def test_sf(self, x, c, ref):
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(stats.genlogistic.sf(x, c), ref, rtol=1e-14)

    # 使用 pytest.mark.parametrize 装饰器，为 test_isf 方法提供参数化测试数据
    # 每组参数包括输入值 q, c 和期望输出值 ref
    @pytest.mark.parametrize('q, c, ref', [(0.01, 200, 9.898441467379765),
                                           (0.001, 2, 7.600152115573173)])
    # 定义测试方法 test_isf，测试 genlogistic 分布的 isf 函数
    def test_isf(self, q, c, ref):
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(stats.genlogistic.isf(q, c), ref, rtol=5e-16)

    # 使用 pytest.mark.parametrize 装饰器，为 test_ppf 方法提供参数化测试数据
    # 每组参数包括输入值 q, c 和期望输出值 ref
    @pytest.mark.parametrize('q, c, ref', [(0.5, 200, 5.6630969187064615),
                                           (0.99, 20, 7.595630231412436)])
    # 定义测试方法 test_ppf，测试 genlogistic 分布的 ppf 函数
    def test_ppf(self, q, c, ref):
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(stats.genlogistic.ppf(q, c), ref, rtol=5e-16)

    # 使用 pytest.mark.parametrize 装饰器，为 test_logcdf 方法提供参数化测试数据
    # 每组参数包括输入值 x, c 和期望输出值 ref
    @pytest.mark.parametrize('x, c, ref', [(100, 0.02, -7.440151952041672e-46),
                                           (50, 20, -3.857499695927835e-21)])
    # 定义测试方法 test_logcdf，测试 genlogistic 分布的 logcdf 函数
    def test_logcdf(self, x, c, ref):
        # 使用 assert_allclose 断言函数来验证计算结果与期望值的接近程度
        assert_allclose(stats.genlogistic.logcdf(x, c), ref, rtol=1e-15)


# 定义一个测试类 TestHypergeom，用于测试超几何分布相关功能
class TestHypergeom:
    # 在每个测试方法运行前，设置随机数种子为 1234，保证测试的可重复性
    def setup_method(self):
        np.random.seed(1234)
    def test_rvs(self):
        # 使用超几何分布生成随机变量
        vals = stats.hypergeom.rvs(20, 10, 3, size=(2, 50))
        # 断言所有生成的值都在指定范围内
        assert np.all(vals >= 0) & np.all(vals <= 3)
        # 断言生成的数组形状符合预期
        assert np.shape(vals) == (2, 50)
        # 断言生成的值类型为整数
        assert vals.dtype.char in typecodes['AllInteger']
        # 使用超几何分布生成单个随机变量
        val = stats.hypergeom.rvs(20, 3, 10)
        # 断言生成的值类型为整数
        assert isinstance(val, int)
        # 使用超几何分布对象生成多个随机变量
        val = stats.hypergeom(20, 3, 10).rvs(3)
        # 断言生成的值类型为 NumPy 数组
        assert isinstance(val, np.ndarray)
        # 断言生成的数组元素类型为整数
        assert val.dtype.char in typecodes['AllInteger']

    def test_precision(self):
        # 使用 mpmath 的比较数字
        M = 2500
        n = 50
        N = 500
        tot = M
        good = n
        # 计算超几何分布的概率质量函数值
        hgpmf = stats.hypergeom.pmf(2, tot, good, N)
        # 断言计算得到的值与预期值接近
        assert_almost_equal(hgpmf, 0.0010114963068932233, 11)

    def test_args(self):
        # 测试参数的边缘情况下是否能得到正确的输出
        # 参见 GitHub 问题编号 gh-2325
        assert_almost_equal(stats.hypergeom.pmf(0, 2, 1, 0), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)

        assert_almost_equal(stats.hypergeom.pmf(0, 2, 0, 2), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)

    def test_cdf_above_one(self):
        # 对于某些参数值，超几何分布的累积分布函数可能大于1，参见 GitHub 问题编号 gh-2238
        assert_(0 <= stats.hypergeom.cdf(30, 13397950, 4363, 12390) <= 1.0)

    def test_precision2(self):
        # 测试超大数值情况下的超几何分布精度，参见 Issue #1218
        # 结果与 R 软件比较
        oranges = 9.9e4
        pears = 1.1e5
        fruits_eaten = np.array([3, 3.8, 3.9, 4, 4.1, 4.2, 5]) * 1e4
        quantile = 2e4
        # 计算超几何分布的生存函数值
        res = [stats.hypergeom.sf(quantile, oranges + pears, oranges, eaten)
               for eaten in fruits_eaten]
        expected = np.array([0, 1.904153e-114, 2.752693e-66, 4.931217e-32,
                             8.265601e-11, 0.1237904, 1])
        # 断言计算得到的值与预期值接近
        assert_allclose(res, expected, atol=0, rtol=5e-7)

        # 使用数组作为第一个参数进行测试
        quantiles = [1.9e4, 2e4, 2.1e4, 2.15e4]
        res2 = stats.hypergeom.sf(quantiles, oranges + pears, oranges, 4.2e4)
        expected2 = [1, 0.1237904, 6.511452e-34, 3.277667e-69]
        # 断言计算得到的值与预期值接近
        assert_allclose(res2, expected2, atol=0, rtol=5e-7)

    def test_entropy(self):
        # 简单测试熵的计算
        hg = stats.hypergeom(4, 1, 1)
        h = hg.entropy()
        expected_p = np.array([0.75, 0.25])
        expected_h = -np.sum(xlogy(expected_p, expected_p))
        # 断言计算得到的熵与预期值接近
        assert_allclose(h, expected_h)

        hg = stats.hypergeom(1, 1, 1)
        h = hg.entropy()
        # 断言计算得到的熵为0
        assert_equal(h, 0.0)
    def test_logsf(self):
        # Test logsf for very large numbers. See issue #4982
        # Results compare with those from R (v3.2.0):
        # phyper(k, n, M-n, N, lower.tail=FALSE, log.p=TRUE)
        # -2239.771

        k = 1e4  # 设置超几何分布的参数 k
        M = 1e7  # 设置超几何分布的参数 M
        n = 1e6  # 设置超几何分布的参数 n
        N = 5e4  # 设置超几何分布的参数 N

        result = stats.hypergeom.logsf(k, M, n, N)  # 计算超几何分布的 logsf
        expected = -2239.771   # 从 R 中获取的期望值
        assert_almost_equal(result, expected, decimal=3)  # 断言结果与期望值接近

        k = 1  # 更新超几何分布的参数 k
        M = 1600  # 更新超几何分布的参数 M
        n = 600  # 更新超几何分布的参数 n
        N = 300  # 更新超几何分布的参数 N

        result = stats.hypergeom.logsf(k, M, n, N)  # 计算超几何分布的 logsf
        expected = -2.566567e-68   # 从 R 中获取的期望值
        assert_almost_equal(result, expected, decimal=15)  # 断言结果与期望值接近

    def test_logcdf(self):
        # Test logcdf for very large numbers. See issue #8692
        # Results compare with those from R (v3.3.2):
        # phyper(k, n, M-n, N, lower.tail=TRUE, log.p=TRUE)
        # -5273.335

        k = 1  # 设置超几何分布的参数 k
        M = 1e7  # 设置超几何分布的参数 M
        n = 1e6  # 设置超几何分布的参数 n
        N = 5e4  # 设置超几何分布的参数 N

        result = stats.hypergeom.logcdf(k, M, n, N)  # 计算超几何分布的 logcdf
        expected = -5273.335   # 从 R 中获取的期望值
        assert_almost_equal(result, expected, decimal=3)  # 断言结果与期望值接近

        # Same example as in issue #8692
        k = 40  # 更新超几何分布的参数 k
        M = 1600  # 更新超几何分布的参数 M
        n = 50  # 更新超几何分布的参数 n
        N = 300  # 更新超几何分布的参数 N

        result = stats.hypergeom.logcdf(k, M, n, N)  # 计算超几何分布的 logcdf
        expected = -7.565148879229e-23    # 从 R 中获取的期望值
        assert_almost_equal(result, expected, decimal=15)  # 断言结果与期望值接近

        k = 125  # 更新超几何分布的参数 k
        M = 1600  # 更新超几何分布的参数 M
        n = 250  # 更新超几何分布的参数 n
        N = 500  # 更新超几何分布的参数 N

        result = stats.hypergeom.logcdf(k, M, n, N)  # 计算超几何分布的 logcdf
        expected = -4.242688e-12    # 从 R 中获取的期望值
        assert_almost_equal(result, expected, decimal=15)  # 断言结果与期望值接近

        # test broadcasting robustness based on reviewer
        # concerns in PR 9603; using an array version of
        # the example from issue #8692
        k = np.array([40, 40, 40])  # 使用数组形式的超几何分布的参数 k
        M = 1600  # 超几何分布的参数 M
        n = 50  # 超几何分布的参数 n
        N = 300  # 超几何分布的参数 N

        result = stats.hypergeom.logcdf(k, M, n, N)  # 计算超几何分布的 logcdf
        expected = np.full(3, -7.565148879229e-23)  # 从 R 中获取的填充期望值
        assert_almost_equal(result, expected, decimal=15)  # 断言结果与期望值接近

    def test_mean_gh18511(self):
        # gh-18511 reported that the `mean` was incorrect for large arguments;
        # check that this is resolved
        M = 390_000  # 设置超几何分布的参数 M
        n = 370_000  # 设置超几何分布的参数 n
        N = 12_000  # 设置超几何分布的参数 N

        hm = stats.hypergeom.mean(M, n, N)  # 计算超几何分布的均值
        rm = n / M * N  # 根据公式计算超几何分布的均值
        assert_allclose(hm, rm)  # 断言计算的均值与期望值接近

    @pytest.mark.xslow
    def test_sf_gh18506(self):
        # gh-18506 reported that `sf` was incorrect for large population;
        # check that this is resolved
        n = 10  # 设置超几何分布的参数 n
        N = 10**5  # 设置超几何分布的参数 N
        i = np.arange(5, 15)  # 创建一个整数数组
        population_size = 10.**i  # 计算人口大小的数组
        p = stats.hypergeom.sf(n - 1, population_size, N, n)  # 计算超几何分布的 sf
        assert np.all(p > 0)  # 断言所有 sf 结果大于零
        assert np.all(np.diff(p) < 0)  # 断言 sf 结果递减
# 定义一个测试类 TestLoggamma，用于测试 loggamma 分布的统计函数
class TestLoggamma:

    # 对于给定的 x 和 c 值，验证累积分布函数的正确性
    # 使用 pytest 的参数化装饰器，列出多组输入和预期的累积分布函数值
    @pytest.mark.parametrize('x, c, cdf',
                             [(1, 2, 0.7546378854206702),
                              (-1, 14, 6.768116452566383e-18),
                              (-745.1, 0.001, 0.4749605142005238),
                              (-800, 0.001, 0.44958802911019136),
                              (-725, 0.1, 3.4301205868273265e-32),
                              (-740, 0.75, 1.0074360436599631e-241)])
    def test_cdf_ppf(self, x, c, cdf):
        # 计算 loggamma 分布的累积分布函数
        p = stats.loggamma.cdf(x, c)
        # 断言计算得到的累积分布函数值与预期值的接近程度
        assert_allclose(p, cdf, rtol=1e-13)
        # 使用累积分布函数的结果计算其反函数，并再次断言结果与输入值 x 的接近程度
        y = stats.loggamma.ppf(cdf, c)
        assert_allclose(y, x, rtol=1e-13)

    # 对于给定的 x 和 c 值，验证生存函数的正确性
    # 使用 pytest 的参数化装饰器，列出多组输入和预期的生存函数值
    @pytest.mark.parametrize('x, c, sf',
                             [(4, 1.5, 1.6341528919488565e-23),
                              (6, 100, 8.23836829202024e-74),
                              (-800, 0.001, 0.5504119708898086),
                              (-743, 0.0025, 0.8437131370024089)])
    def test_sf_isf(self, x, c, sf):
        # 计算 loggamma 分布的生存函数
        s = stats.loggamma.sf(x, c)
        # 断言计算得到的生存函数值与预期值的接近程度
        assert_allclose(s, sf, rtol=1e-13)
        # 使用生存函数的结果计算其反函数，并再次断言结果与输入值 x 的接近程度
        y = stats.loggamma.isf(sf, c)
        assert_allclose(y, x, rtol=1e-13)

    # 测试 loggamma 分布的对数概率密度函数
    def test_logpdf(self):
        # 对于特定的 x=-500, c=2，预期 ln(gamma(2)) = 0，并且 exp(-500) 约为 7e-218，
        # 远小于 c*x=-1000 的 ULP，因此 logpdf(-500, 2) 应该接近于 -1000.0
        lp = stats.loggamma.logpdf(-500, 2)
        # 断言计算得到的对数概率密度函数值与预期值的接近程度
        assert_allclose(lp, -1000.0, rtol=1e-14)

    # 测试 loggamma 分布的统计量
    def test_stats(self):
        # 下面的预计算值来自 "A Statistical Study of Log-Gamma Distribution" 中的表格
        # 验证 loggamma 分布的均值、方差、偏度和峰度计算的准确性
        table = np.array([
                # c,    mean,   var,    skew,    exc. kurt.
                0.5, -1.9635, 4.9348, -1.5351, 4.0000,
                1.0, -0.5772, 1.6449, -1.1395, 2.4000,
                12.0, 2.4427, 0.0869, -0.2946, 0.1735,
            ]).reshape(-1, 5)
        for c, mean, var, skew, kurt in table:
            # 计算 loggamma 分布的统计量，并断言计算结果与预期值的接近程度
            computed = stats.loggamma.stats(c, moments='msvk')
            assert_array_almost_equal(computed, [mean, var, skew, kurt],
                                      decimal=4)

    # 参数化测试，验证当 c 取特定值时的 loggamma 分布
    @pytest.mark.parametrize('c', [0.1, 0.001])
    # 定义一个测试函数，用于验证 gh-11094 的回归测试。
    def test_rvs(self, c):
        # 使用 loggamma 分布生成指定参数 c 下的随机样本，大小为 100000
        x = stats.loggamma.rvs(c, size=100000)
        # 在 gh-11094 修复之前，当 c=0.001 时会生成许多 -inf 的值。
        assert np.isfinite(x).all()
        # 简单的统计测试。大约一半的值应该小于中位数，另一半大于中位数。
        med = stats.loggamma.median(c)
        # 进行二项分布检验，检验 x 中小于中位数的值的比例是否显著不同于 0.5
        btest = stats.binomtest(np.count_nonzero(x < med), len(x))
        # 计算置信区间
        ci = btest.proportion_ci(confidence_level=0.999)
        # 断言置信区间的下限小于 0.5，上限大于 0.5
        assert ci.low < 0.5 < ci.high

    # 使用参数化测试框架 pytest.mark.parametrize 注解这个测试函数
    @pytest.mark.parametrize("c, ref",
                             [(1e-8, 19.420680753952364),
                              (1, 1.5772156649015328),
                              (1e4, -3.186214986116763),
                              (1e10, -10.093986931748889),
                              (1e100, -113.71031611649761)])
    # 定义计算熵的测试函数
    def test_entropy(self, c, ref):

        # 参考值是使用 mpmath 计算得到的
        # from mpmath import mp
        # mp.dps = 500
        # def loggamma_entropy_mpmath(c):
        #     c = mp.mpf(c)
        #     return float(mp.log(mp.gamma(c)) + c * (mp.one - mp.digamma(c)))

        # 使用 assert_allclose 断言 loggamma.entropy(c) 的返回值与参考值 ref 接近
        assert_allclose(stats.loggamma.entropy(c), ref, rtol=1e-14)
class TestJohnsonsu:
    # reference values were computed via mpmath
    # 引用值是通过 mpmath 计算得出的
    # from mpmath import mp
    # mp.dps = 50
    # def johnsonsu_sf(x, a, b):
    #     x = mp.mpf(x)
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     return float(mp.ncdf(-(a + b * mp.log(x + mp.sqrt(x*x + 1)))))
    # 上述代码段中的函数和精度设定
    # Order is x, a, b, sf, isf tol
    # (Can't expect full precision when the ISF input is very nearly 1)
    # 次序为 x, a, b, sf, isf 的公差
    cases = [(-500, 1, 1, 0.9999999982660072, 1e-8),
             (2000, 1, 1, 7.426351000595343e-21, 5e-14),
             (100000, 1, 1, 4.046923979269977e-40, 5e-14)]

    @pytest.mark.parametrize("case", cases)
    def test_sf_isf(self, case):
        x, a, b, sf, tol = case
        assert_allclose(stats.johnsonsu.sf(x, a, b), sf, rtol=5e-14)
        assert_allclose(stats.johnsonsu.isf(sf, a, b), x, rtol=tol)


class TestJohnsonb:
    # reference values were computed via mpmath
    # 引用值是通过 mpmath 计算得出的
    # from mpmath import mp
    # mp.dps = 50
    # def johnsonb_sf(x, a, b):
    #     x = mp.mpf(x)
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     return float(mp.ncdf(-(a + b * mp.log(x/(mp.one - x)))))
    # 上述代码段中的函数和精度设定
    # Order is x, a, b, sf, isf atol
    # (Can't expect full precision when the ISF input is very nearly 1)
    # 次序为 x, a, b, sf, isf 的绝对公差
    cases = [(1e-4, 1, 1, 0.9999999999999999, 1e-7),
             (0.9999, 1, 1, 8.921114313932308e-25, 5e-14),
             (0.999999, 1, 1, 5.815197487181902e-50, 5e-14)]

    @pytest.mark.parametrize("case", cases)
    def test_sf_isf(self, case):
        x, a, b, sf, tol = case
        assert_allclose(stats.johnsonsb.sf(x, a, b), sf, rtol=5e-14)
        assert_allclose(stats.johnsonsb.isf(sf, a, b), x, atol=tol)


class TestLogistic:
    # gh-6226
    # 测试 Logistic 分布的累积分布函数和反函数
    def test_cdf_ppf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.cdf(x)
        xx = stats.logistic.ppf(y)
        assert_allclose(x, xx)

    # 测试 Logistic 分布的生存函数和反生存函数
    def test_sf_isf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.sf(x)
        xx = stats.logistic.isf(y)
        assert_allclose(x, xx)

    # 测试 Logistic 分布在极端值上的行为
    def test_extreme_values(self):
        # p is chosen so that 1 - (1 - p) == p in double precision
        # 选择 p 值使得在双精度下 1 - (1 - p) == p
        p = 9.992007221626409e-16
        desired = 34.53957599234088
        assert_allclose(stats.logistic.ppf(1 - p), desired)
        assert_allclose(stats.logistic.isf(p), desired)

    # 测试 Logistic 分布的对数概率密度函数
    def test_logpdf_basic(self):
        logp = stats.logistic.logpdf([-15, 0, 10])
        # Expected values computed with mpmath with 50 digits of precision.
        # 使用 mpmath 计算的预期值，精度为 50 位小数
        expected = [-15.000000611804547,
                    -1.3862943611198906,
                    -10.000090797798434]
        assert_allclose(logp, expected, rtol=1e-13)

    # 测试 Logistic 分布在极端值上的对数概率密度函数
    def test_logpdf_extreme_values(self):
        logp = stats.logistic.logpdf([800, -800])
        # For such large arguments, logpdf(x) = -abs(x) when computed
        # with 64 bit floating point.
        # 对于如此大的参数，使用 64 位浮点数计算时，logpdf(x) = -abs(x)
        assert_equal(logp, [-800, -800])
    # 使用 pytest 的参数化装饰器，为 test_fit 方法提供多组参数
    @pytest.mark.parametrize("loc_rvs,scale_rvs", [(0.4484955, 0.10216821),
                                                   (0.62918191, 0.74367064)])
    # 定义测试方法 test_fit，接受 loc_rvs 和 scale_rvs 作为参数
    def test_fit(self, loc_rvs, scale_rvs):
        # 生成服从 logistic 分布的随机数据
        data = stats.logistic.rvs(size=100, loc=loc_rvs, scale=scale_rvs)

        # 定义一个函数 func，用于计算优化后的结果
        def func(input, data):
            # 解包输入参数
            a, b = input
            # 数据长度
            n = len(data)
            # 计算两个方程的结果
            x1 = np.sum(np.exp((data - a) / b) /
                        (1 + np.exp((data - a) / b))) - n / 2
            x2 = np.sum(((data - a) / b) *
                        ((np.exp((data - a) / b) - 1) /
                         (np.exp((data - a) / b) + 1))) - n
            return x1, x2

        # 使用 root 函数求解 func 函数的根，作为预期解
        expected_solution = root(func, stats.logistic._fitstart(data), args=(
            data,)).x
        # 使用 logistic 分布的 fit 方法拟合数据
        fit_method = stats.logistic.fit(data)

        # 断言拟合方法的结果与预期解在非常小的误差范围内相等
        assert_allclose(fit_method, expected_solution, atol=1e-30)

    # 定义测试方法 test_fit_comp_optimizer，用于测试 logistic 分布的拟合方法
    def test_fit_comp_optimizer(self):
        # 生成服从 logistic 分布的随机数据
        data = stats.logistic.rvs(size=100, loc=0.5, scale=2)
        # 调用 _assert_less_or_close_loglike 函数，验证拟合结果
        _assert_less_or_close_loglike(stats.logistic, data)
        # 另一组参数调用 _assert_less_or_close_loglike 函数，验证拟合结果
        _assert_less_or_close_loglike(stats.logistic, data, floc=1)
        # 另一组参数调用 _assert_less_or_close_loglike 函数，验证拟合结果
        _assert_less_or_close_loglike(stats.logistic, data, fscale=1)

    # 使用 pytest 的参数化装饰器，为 test_logcdfsf_tails 方法提供参数
    @pytest.mark.parametrize('testlogcdf', [True, False])
    # 定义测试方法 test_logcdfsf_tails，测试 logistic 分布的 logcdf 和 logsf 方法
    def test_logcdfsf_tails(self, testlogcdf):
        # 定义一组测试数据 x
        x = np.array([-10000, -800, 17, 50, 500])
        # 根据 testlogcdf 的值选择调用 logcdf 或 logsf 方法
        if testlogcdf:
            y = stats.logistic.logcdf(x)
        else:
            y = stats.logistic.logsf(-x)
        # 预期值是使用 mpmath 计算得到的
        expected = [-10000.0, -800.0, -4.139937633089748e-08,
                    -1.9287498479639178e-22, -7.124576406741286e-218]
        # 断言计算结果与预期值在非常小的误差范围内相等
        assert_allclose(y, expected, rtol=2e-15)

    # 定义测试方法 test_fit_gh_18176，验证 logistic 分布在特定情况下的修复
    def test_fit_gh_18176(self):
        # 给定特定的数据数组，用于验证 logistic.fit 方法的修复
        data = np.array([-459, 37, 43, 45, 45, 48, 54, 55, 58]
                        + [59] * 3 + [61] * 9)
        # 调用 _assert_less_or_close_loglike 函数，验证拟合结果
        _assert_less_or_close_loglike(stats.logistic, data)
class TestLogser:
    # 设置测试环境的随机种子
    def setup_method(self):
        np.random.seed(1234)

    # 测试 stats.logser.rvs 方法
    def test_rvs(self):
        # 生成服从对数级数分布的随机变量
        vals = stats.logser.rvs(0.75, size=(2, 50))
        # 断言所有随机变量都大于等于1
        assert np.all(vals >= 1)
        # 断言随机变量的形状为 (2, 50)
        assert np.shape(vals) == (2, 50)
        # 断言随机变量的数据类型为整数
        assert vals.dtype.char in typecodes['AllInteger']
        # 生成一个服从对数级数分布的单个随机变量，并断言其为整数类型
        val = stats.logser.rvs(0.75)
        assert isinstance(val, int)
        # 生成多个服从对数级数分布的随机变量，并断言返回的是 NumPy 数组
        val = stats.logser(0.75).rvs(3)
        assert isinstance(val, np.ndarray)
        # 断言返回的随机变量数组的数据类型为整数
        assert val.dtype.char in typecodes['AllInteger']

    # 测试 stats.logser.pmf 方法在小概率情况下的概率质量函数
    def test_pmf_small_p(self):
        # 计算概率质量函数在给定参数下的数值
        m = stats.logser.pmf(4, 1e-20)
        # 使用 mpmath 计算的预期值，以验证计算结果的正确性
        assert_allclose(m, 2.5e-61)

    # 测试 stats.logser.mean 方法在小概率情况下的均值计算
    def test_mean_small_p(self):
        # 计算均值在给定参数下的数值
        m = stats.logser.mean(1e-8)
        # 使用 mpmath 计算的预期均值，以验证计算结果的正确性
        assert_allclose(m, 1.000000005)


class TestGumbel_r_l:
    # 定义测试用例级别的 fixture，用于生成随机数生成器
    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    # 参数化测试，分别对 Gumbel 右尾和左尾分布进行拟合测试
    @pytest.mark.parametrize("dist", [stats.gumbel_r, stats.gumbel_l])
    @pytest.mark.parametrize("loc_rvs", [-1, 0, 1])
    @pytest.mark.parametrize("scale_rvs", [.1, 1, 5])
    @pytest.mark.parametrize('fix_loc, fix_scale',
                             ([True, False], [False, True]))
    def test_fit_comp_optimizer(self, dist, loc_rvs, scale_rvs,
                                fix_loc, fix_scale, rng):
        # 生成符合 Gumbel 分布的随机数据
        data = dist.rvs(size=100, loc=loc_rvs, scale=scale_rvs,
                        random_state=rng)

        kwds = dict()
        # 如果需要固定位置或者尺度参数，修改关键字参数
        if fix_loc:
            kwds['floc'] = loc_rvs * 2
        if fix_scale:
            kwds['fscale'] = scale_rvs * 2

        # 调用自定义的 _assert_less_or_close_loglike 方法，验证拟合效果
        _assert_less_or_close_loglike(dist, data, **kwds)

    # 参数化测试，分别对 Gumbel 右尾和左尾分布进行拟合验证
    @pytest.mark.parametrize("dist, sgn", [(stats.gumbel_r, 1),
                                           (stats.gumbel_l, -1)])
    def test_fit(self, dist, sgn):
        # 构造测试数据
        z = sgn*np.array([3, 3, 3, 3, 3, 3, 3, 3.00000001])
        # 进行分布拟合，获取估计的位置参数和尺度参数
        loc, scale = dist.fit(z)
        # 使用 mpmath 计算的预期值，以验证拟合结果的正确性
        assert_allclose(loc, sgn*3.0000000001667906)
        assert_allclose(scale, 1.2495222465145514e-09, rtol=1e-6)


class TestPareto:
    def test_stats(self):
        # 测试 stats() 方法，使用一些简单的值进行检查。同时确保计算过程中不会触发 RuntimeWarnings。
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            # 使用 Pareto 分布计算指定参数下的统计量，并进行断言检查
            m, v, s, k = stats.pareto.stats(0.5, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(1.0, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(1.5, moments='mvsk')
            assert_equal(m, 3.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(2.0, moments='mvsk')
            assert_equal(m, 2.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(2.5, moments='mvsk')
            assert_allclose(m, 2.5 / 1.5)
            assert_allclose(v, 2.5 / (1.5*1.5*0.5))
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(3.0, moments='mvsk')
            assert_allclose(m, 1.5)
            assert_allclose(v, 0.75)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(3.5, moments='mvsk')
            assert_allclose(m, 3.5 / 2.5)
            assert_allclose(v, 3.5 / (2.5*2.5*1.5))
            assert_allclose(s, (2*4.5/0.5)*np.sqrt(1.5/3.5))
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(4.0, moments='mvsk')
            assert_allclose(m, 4.0 / 3.0)
            assert_allclose(v, 4.0 / 18.0)
            assert_allclose(s, 2*(1+4.0)/(4.0-3) * np.sqrt((4.0-2)/4.0))
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(4.5, moments='mvsk')
            assert_allclose(m, 4.5 / 3.5)
            assert_allclose(v, 4.5 / (3.5*3.5*2.5))
            assert_allclose(s, (2*5.5/1.5) * np.sqrt(2.5/4.5))
            assert_allclose(k, 6*(4.5**3 + 4.5**2 - 6*4.5 - 2)/(4.5*1.5*0.5))

    def test_sf(self):
        # 测试 pareto.sf() 方法的功能
        x = 1e9
        b = 2
        scale = 1.5
        # 计算 Pareto 分布的生存函数，并进行断言检查
        p = stats.pareto.sf(x, b, loc=0, scale=scale)
        expected = (scale/x)**b   # 预期结果
        assert_allclose(p, expected)

    @pytest.fixture(scope='function')
    def rng(self):
        # 返回一个使用种子为 1234 的默认随机数生成器对象作为测试的 fixture
        return np.random.default_rng(1234)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in "
                                "double_scalars")
    @pytest.mark.parametrize("rvs_shape", [1, 2])
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 5])
    # 使用参数化测试标记，分别测试不同参数下的随机变量生成
    @pytest.mark.parametrize("rvs_shape", [.1, 2])
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [p for p in product([True, False], repeat=3)
                              if False in p])
    @np.errstate(invalid="ignore")
    # 使用 pytest 的参数化装饰器定义多组参数来测试 `test_fit_MLE_comp_optimizer` 方法
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        # 生成服从 Pareto 分布的随机数据
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale,
                                loc=rvs_loc, random_state=rng)

        kwds = {}
        # 根据参数设置 `kwds` 字典，用于传递给 `stats.pareto.fit` 函数
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 调用自定义的 `_assert_less_or_close_loglike` 函数，验证 Pareto 拟合结果的对数似然
        _assert_less_or_close_loglike(stats.pareto, data, **kwds)

    @np.errstate(invalid="ignore")
    # 测试对已知不良种子的情况
    def test_fit_known_bad_seed(self):
        # 设定参数
        shape, location, scale = 1, 0, 1
        # 使用特定种子生成 Pareto 分布数据
        data = stats.pareto.rvs(shape, location, scale, size=100,
                                random_state=np.random.default_rng(2535619))
        # 调用自定义的 `_assert_less_or_close_loglike` 函数，验证 Pareto 拟合结果的对数似然
        _assert_less_or_close_loglike(stats.pareto, data)

    # 测试 Pareto 拟合中的警告情况
    def test_fit_warnings(self):
        # 检查 Pareto 拟合是否产生警告
        assert_fit_warnings(stats.pareto)
        # 测试 `floc` 参数设置为正数时是否引发 `FitDataError` 异常
        assert_raises(FitDataError, stats.pareto.fit, [1, 2, 3], floc=2)
        # 测试 `floc` 和 `fscale` 参数组合是否引发 `FitDataError` 异常
        assert_raises(FitDataError, stats.pareto.fit, [5, 2, 3], floc=1,
                      fscale=3)
    # 定义一个测试方法，用于测试负数据的情况，接受一个随机数生成器对象 rng 作为参数
    def test_negative_data(self, rng):
        # 使用 Pareto 分布生成包含 100 个数据点的数据集，参数 loc=-130 表示偏移，b=1 表示形状参数
        # random_state=rng 用于控制随机数生成的状态
        data = stats.pareto.rvs(loc=-130, b=1, size=100, random_state=rng)
        
        # 断言：验证数据集中所有数据都小于 0
        assert_array_less(data, 0)
        
        # 此测试的目的是确保对所有负数据不会引发运行时警告，而不是检查 fit 方法的输出。
        # 其他方法测试输出，但必须消除超类方法产生的警告。
        # 调用 Pareto 分布的 fit 方法拟合数据，将返回的结果赋值给变量 _
        _ = stats.pareto.fit(data)
class TestGenpareto:
    # 定义测试类 TestGenpareto
    def test_ab(self):
        # 测试方法 test_ab，用于测试 genpareto 分布中 a 和 b 的取值
        # c >= 0: a, b = [0, inf]
        for c in [1., 0.]:
            # 将 c 转换为 NumPy 数组
            c = np.asarray(c)
            # 调用 _get_support 方法获取 a 和 b 的取值
            a, b = stats.genpareto._get_support(c)
            # 断言 a 应该等于 0
            assert_equal(a, 0.)
            # 断言 b 应为正无穷
            assert_(np.isposinf(b))

        # c < 0: a=0, b=1/|c|
        c = np.asarray(-2.)
        # 调用 _get_support 方法获取 a 和 b 的取值
        a, b = stats.genpareto._get_support(c)
        # 断言 a 应该等于 0
        assert_allclose([a, b], [0., 0.5])

    def test_c0(self):
        # 测试方法 test_c0，验证当 c=0 时，genpareto 分布退化为指数分布
        # rv = stats.genpareto(c=0.)
        rv = stats.genpareto(c=0.)
        # 在区间 [0, 10] 上生成 30 个均匀分布的点作为输入
        x = np.linspace(0, 10., 30)
        # 断言 genpareto 分布的概率密度函数值与指数分布的值非常接近
        assert_allclose(rv.pdf(x), stats.expon.pdf(x))
        # 断言 genpareto 分布的累积分布函数值与指数分布的值非常接近
        assert_allclose(rv.cdf(x), stats.expon.cdf(x))
        # 断言 genpareto 分布的生存函数值与指数分布的值非常接近
        assert_allclose(rv.sf(x), stats.expon.sf(x))

        # 在区间 [0, 1] 上生成 10 个均匀分布的点作为输入
        q = np.linspace(0., 1., 10)
        # 断言 genpareto 分布的百分点函数值与指数分布的值非常接近
        assert_allclose(rv.ppf(q), stats.expon.ppf(q))

    def test_cm1(self):
        # 测试方法 test_cm1，验证当 c=-1 时，genpareto 分布退化为均匀分布在 [0, 1] 上
        rv = stats.genpareto(c=-1.)
        # 在区间 [0, 10] 上生成 30 个均匀分布的点作为输入
        x = np.linspace(0, 10., 30)
        # 断言 genpareto 分布的概率密度函数值与均匀分布的值非常接近
        assert_allclose(rv.pdf(x), stats.uniform.pdf(x))
        # 断言 genpareto 分布的累积分布函数值与均匀分布的值非常接近
        assert_allclose(rv.cdf(x), stats.uniform.cdf(x))
        # 断言 genpareto 分布的生存函数值与均匀分布的值非常接近
        assert_allclose(rv.sf(x), stats.uniform.sf(x))

        # 断言 logpdf(1., c=-1) 应该为零
        assert_allclose(rv.logpdf(1), 0)

    def test_x_inf(self):
        # 测试方法 test_x_inf，确保当 x=inf 时能够优雅地处理
        rv = stats.genpareto(c=0.1)
        # 断言 genpareto 分布在 x=inf 时的概率密度函数值和累积分布函数值分别为 0 和 1
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        # 断言 genpareto 分布在 x=inf 时的对数概率密度函数值为负无穷
        assert_(np.isneginf(rv.logpdf(np.inf)))

        rv = stats.genpareto(c=0.)
        # 断言 genpareto 分布在 x=inf 时的概率密度函数值和累积分布函数值分别为 0 和 1
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        # 断言 genpareto 分布在 x=inf 时的对数概率密度函数值为负无穷
        assert_(np.isneginf(rv.logpdf(np.inf)))

        rv = stats.genpareto(c=-1.)
        # 断言 genpareto 分布在 x=inf 时的概率密度函数值和累积分布函数值分别为 0 和 1
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        # 断言 genpareto 分布在 x=inf 时的对数概率密度函数值为负无穷
        assert_(np.isneginf(rv.logpdf(np.inf)))

    def test_c_continuity(self):
        # 测试方法 test_c_continuity，验证 genpareto 分布在 c=0 和 c=-1 时的概率密度函数和累积分布函数的连续性
        x = np.linspace(0, 10, 30)
        for c in [0, -1]:
            pdf0 = stats.genpareto.pdf(x, c)
            for dc in [1e-14, -1e-14]:
                pdfc = stats.genpareto.pdf(x, c + dc)
                # 断言在极小的 dc 变化下，pdf 的值保持不变
                assert_allclose(pdf0, pdfc, atol=1e-12)

            cdf0 = stats.genpareto.cdf(x, c)
            for dc in [1e-14, 1e-14]:
                cdfc = stats.genpareto.cdf(x, c + dc)
                # 断言在极小的 dc 变化下，cdf 的值保持不变
                assert_allclose(cdf0, cdfc, atol=1e-12)

    def test_c_continuity_ppf(self):
        # 测试方法 test_c_continuity_ppf，验证 genpareto 分布在 c=0 和 c=-1 时的百分点函数的连续性
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [0., -1.]:
            ppf0 = stats.genpareto.ppf(q, c)
            for dc in [1e-14, -1e-14]:
                ppfc = stats.genpareto.ppf(q, c + dc)
                # 断言在极小的 dc 变化下，ppf 的值保持不变
                assert_allclose(ppf0, ppfc, atol=1e-12)
    # 定义一个测试方法，用于验证连续性参数 c 对应的逆生存函数是否连续
    def test_c_continuity_isf(self):
        # 生成一个测试用的分位数数组 q，包括对数空间、线性空间和逆序对数空间
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        # 对于每个 c 的值，执行以下测试
        for c in [0., -1.]:
            # 计算 c 对应的逆生存函数值 isf0
            isf0 = stats.genpareto.isf(q, c)
            # 对于每个微小变化 dc，执行以下测试
            for dc in [1e-14, -1e-14]:
                # 计算 c+dc 对应的逆生存函数值 isfc
                isfc = stats.genpareto.isf(q, c + dc)
                # 验证 isf0 和 isfc 是否在指定的容差范围内相等
                assert_allclose(isf0, isfc, atol=1e-12)

    # 定义一个测试方法，验证累积分布函数和百分点函数之间的回路一致性
    def test_cdf_ppf_roundtrip(self):
        # 生成一个测试用的分位数数组 q，包括对数空间、线性空间和逆序对数空间
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        # 对于每个 c 的值，执行以下测试
        for c in [1e-8, -1e-18, 1e-15, -1e-15]:
            # 验证累积分布函数和百分点函数之间的一致性
            assert_allclose(stats.genpareto.cdf(stats.genpareto.ppf(q, c), c),
                            q, atol=1e-15)

    # 定义一个测试方法，验证对数生存函数的计算是否正确
    def test_logsf(self):
        # 计算给定参数下的对数生存函数值 logp
        logp = stats.genpareto.logsf(1e10, .01, 0, 1)
        # 验证计算得到的 logp 是否与预期值相等
        assert_allclose(logp, -1842.0680753952365)

    # 使用参数化测试装饰器，验证统计量（均值、方差、偏度、超额峰度）的计算是否准确
    @pytest.mark.parametrize(
        'c, expected_stats',
        [(0, [1, 1, 2, 6]),
         (1/4, [4/3, 32/9, 10/np.sqrt(2), np.nan]),
         (1/9, [9/8, (81/64)*(9/7), (10/9)*np.sqrt(7), 754/45]),
         (-1, [1/2, 1/12, 0, -6/5])])
    def test_stats(self, c, expected_stats):
        # 计算给定 c 下的统计量（均值、方差、偏度、超额峰度）
        result = stats.genpareto.stats(c, moments='mvsk')
        # 验证计算得到的统计量 result 是否与预期的 expected_stats 相等
        assert_allclose(result, expected_stats, rtol=1e-13, atol=1e-15)

    # 定义一个测试方法，验证方差计算的准确性
    def test_var(self):
        # 执行针对 gh-11168 的回归测试，验证给定参数下的方差计算是否正确
        v = stats.genpareto.var(1e-8)
        # 验证计算得到的方差 v 是否与预期值相等
        assert_allclose(v, 1.000000040000001, rtol=1e-13)
# 定义一个测试类 TestPearson3，用于测试 scipy.stats 中 Pearson Type III 分布的函数
class TestPearson3:
    
    # 在每个测试方法执行前，设置随机种子，保证测试结果可复现
    def setup_method(self):
        np.random.seed(1234)

    # 测试 rvs 方法，生成符合 Pearson Type III 分布的随机变量，并进行断言检查
    def test_rvs(self):
        # 生成指定参数下的随机变量数组
        vals = stats.pearson3.rvs(0.1, size=(2, 50))
        # 断言生成的数组形状符合预期
        assert np.shape(vals) == (2, 50)
        # 断言生成的数组元素类型为浮点数
        assert vals.dtype.char in typecodes['AllFloat']
        # 生成单个随机变量，并断言其类型为浮点数
        val = stats.pearson3.rvs(0.5)
        assert isinstance(val, float)
        # 使用指定参数生成多个随机变量数组，并断言其类型为 NumPy 数组
        val = stats.pearson3(0.5).rvs(3)
        assert isinstance(val, np.ndarray)
        # 断言生成的数组元素类型为浮点数
        assert val.dtype.char in typecodes['AllFloat']
        # 断言生成的数组长度符合预期
        assert len(val) == 3

    # 测试 pdf 方法，计算 Pearson Type III 分布的概率密度函数值，并进行断言检查
    def test_pdf(self):
        # 计算指定参数和变量值下的概率密度函数值，并断言结果与预期值在指定精度范围内相等
        vals = stats.pearson3.pdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.05399097, 0.05555481, 0.05670246]),
                        atol=1e-6)
        vals = stats.pearson3.pdf(-3, 0.1)
        assert_allclose(vals, np.array([0.00313791]), atol=1e-6)
        vals = stats.pearson3.pdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, np.array([0.00313791, 0.05192304, 0.25028092,
                                        0.39885918, 0.23413173]), atol=1e-6)

    # 测试 cdf 方法，计算 Pearson Type III 分布的累积分布函数值，并进行断言检查
    def test_cdf(self):
        # 计算指定参数和变量值下的累积分布函数值，并断言结果与预期值在指定精度范围内相等
        vals = stats.pearson3.cdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.97724987, 0.97462004, 0.97213626]),
                        atol=1e-6)
        vals = stats.pearson3.cdf(-3, 0.1)
        assert_allclose(vals, [0.00082256], atol=1e-6)
        vals = stats.pearson3.cdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, [8.22563821e-04, 1.99860448e-02, 1.58550710e-01,
                               5.06649130e-01, 8.41442111e-01], atol=1e-6)

    # 测试修复了 GitHub issue #11186 中关于负偏度的 CDF 错误
    # 同时检查 Pearson Type III 分布函数的向量化处理
    def test_negative_cdf_bug_11186(self):
        skews = [-3, -1, 0, 0.5]
        x_eval = 0.5
        neg_inf = -30  # 避免由 np.log(0) 引起的 RuntimeWarning
        # 计算指定参数和变量值下的累积分布函数值，并进行断言检查
        cdfs = stats.pearson3.cdf(x_eval, skews)
        # 使用数值积分计算概率密度函数在指定区间内的积分值，并进行断言检查
        int_pdfs = [quad(stats.pearson3(skew).pdf, neg_inf, x_eval)[0]
                    for skew in skews]
        assert_allclose(cdfs, int_pdfs)

    # 测试修复了 GitHub issue #11746 中 pearson3.moment 返回数组长度 0 或 1 的问题
    # 首个矩等于默认为零的位置参数
    def test_return_array_bug_11746(self):
        # 计算 Pearson Type III 分布的指定阶矩，并进行断言检查
        moment = stats.pearson3.moment(1, 2)
        assert_equal(moment, 0)
        # 断言返回的矩是 NumPy 数字类型
        assert isinstance(moment, np.number)

        moment = stats.pearson3.moment(1, 0.000001)
        assert_equal(moment, 0)
        assert isinstance(moment, np.number)
    # 定义一个测试函数，用于检验 gh-17050 中报告的负偏斜度下的 PPF 是否正确
    def test_ppf_bug_17050(self):
        # 不正确的负偏斜度导致了 gh-17050 中的报告
        # 检查此问题是否已修复（即使在数组情况下）
        
        # 定义负偏斜度数组
        skews = [-3, -1, 0, 0.5]
        # 定义评估点
        x_eval = 0.5
        # 计算 PPF 值，使用 Pearson3 分布的累积分布函数和给定的负偏斜度
        res = stats.pearson3.ppf(stats.pearson3.cdf(x_eval, skews), skews)
        # 断言计算的 PPF 值与预期的评估点 x_eval 接近
        assert_allclose(res, x_eval)

        # 负偏斜度的取反会使分布相对于原点翻转，因此以下断言应成立
        # 定义一个包含负偏斜度的 NumPy 数组
        skew = np.array([[-0.5], [1.5]])
        # 在区间 [-2, 2] 上生成一组均匀分布的点
        x = np.linspace(-2, 2)
        # 断言对称性：负偏斜度 skew 和 -skew 的概率密度函数在对称点 x 处的值应接近
        assert_allclose(stats.pearson3.pdf(x, skew),
                        stats.pearson3.pdf(-x, -skew))
        # 断言对称性：负偏斜度 skew 和 -skew 的累积分布函数在对称点 x 处的值应接近
        assert_allclose(stats.pearson3.cdf(x, skew),
                        stats.pearson3.sf(-x, -skew))
        # 断言对称性：负偏斜度 skew 和 -skew 的百分位点函数在对称点 x 处的值应接近
        assert_allclose(stats.pearson3.ppf(x, skew),
                        -stats.pearson3.isf(x, -skew))

    # 定义另一个测试函数，用于检验累积分布函数的正确性
    def test_sf(self):
        # 参考值是通过参考分布计算的，例如 mp.dps = 50; Pearson3(skew=skew).sf(x)。
        # 检查正、负和零偏斜度由于分支的影响。
        
        # 定义偏斜度数组
        skew = [0.1, 0.5, 1.0, -0.1]
        # 定义不同的评估点 x
        x = [5.0, 10.0, 50.0, 8.0]
        # 参考值列表
        ref = [1.64721926440872e-06, 8.271911573556123e-11,
               1.3149506021756343e-40, 2.763057937820296e-21]
        # 断言累积分布函数的计算结果与参考值在指定的相对容差范围内接近
        assert_allclose(stats.pearson3.sf(x, skew), ref, rtol=2e-14)
        # 断言在偏斜度为 0 时，Pearson3 分布的累积分布函数与正态分布的累积分布函数在指定的相对容差范围内接近
        assert_allclose(stats.pearson3.sf(x, 0), stats.norm.sf(x), rtol=2e-14)
class TestKappa4:
    def test_cdf_genpareto(self):
        # 定义测试用例：generalized Pareto 分布，其中 h = 1 且 k != 0
        x = [0.0, 0.1, 0.2, 0.5]
        h = 1.0
        for k in [-1.9, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 1.9]:
            # 计算 kappa4 分布的累积分布函数值
            vals = stats.kappa4.cdf(x, h, k)
            # 生成 Pareto 分布的累积分布函数值，但形状参数相反
            vals_comp = stats.genpareto.cdf(x, -k)
            # 断言两个分布函数值接近
            assert_allclose(vals, vals_comp)

    def test_cdf_genextreme(self):
        # 定义测试用例：generalized extreme value 分布，其中 h = 0 且 k != 0
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = np.linspace(-3, 3, 10)
        # 计算 kappa4 分布的累积分布函数值
        vals = stats.kappa4.cdf(x, h, k)
        # 生成 generalized extreme value 分布的累积分布函数值
        vals_comp = stats.genextreme.cdf(x, k)
        # 断言两个分布函数值接近
        assert_allclose(vals, vals_comp)

    def test_cdf_expon(self):
        # 定义测试用例：指数分布，其中 h = 1 且 k = 0
        x = np.linspace(0, 10, 10)
        h = 1.0
        k = 0.0
        # 计算 kappa4 分布的累积分布函数值
        vals = stats.kappa4.cdf(x, h, k)
        # 生成指数分布的累积分布函数值
        vals_comp = stats.expon.cdf(x)
        # 断言两个分布函数值接近
        assert_allclose(vals, vals_comp)

    def test_cdf_gumbel_r(self):
        # 定义测试用例：右侧 Gumbel 分布，其中 h = 0 且 k = 0
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = 0.0
        # 计算 kappa4 分布的累积分布函数值
        vals = stats.kappa4.cdf(x, h, k)
        # 生成右侧 Gumbel 分布的累积分布函数值
        vals_comp = stats.gumbel_r.cdf(x)
        # 断言两个分布函数值接近
        assert_allclose(vals, vals_comp)

    def test_cdf_logistic(self):
        # 定义测试用例：逻辑斯蒂分布，其中 h = -1 且 k = 0
        x = np.linspace(-5, 5, 10)
        h = -1.0
        k = 0.0
        # 计算 kappa4 分布的累积分布函数值
        vals = stats.kappa4.cdf(x, h, k)
        # 生成逻辑斯蒂分布的累积分布函数值
        vals_comp = stats.logistic.cdf(x)
        # 断言两个分布函数值接近
        assert_allclose(vals, vals_comp)

    def test_cdf_uniform(self):
        # 定义测试用例：均匀分布，其中 h = 1 且 k = 1
        x = np.linspace(-5, 5, 10)
        h = 1.0
        k = 1.0
        # 计算 kappa4 分布的累积分布函数值
        vals = stats.kappa4.cdf(x, h, k)
        # 生成均匀分布的累积分布函数值
        vals_comp = stats.uniform.cdf(x)
        # 断言两个分布函数值接近
        assert_allclose(vals, vals_comp)

    def test_integers_ctor(self):
        # 回归测试：针对整数 h 和 k 的 _argcheck 失败，适用于 numpy 1.12
        stats.kappa4(1, 2)


class TestPoisson:
    def setup_method(self):
        np.random.seed(1234)

    def test_pmf_basic(self):
        # 基本情况测试
        ln2 = np.log(2)
        # 计算泊松分布的概率质量函数值
        vals = stats.poisson.pmf([0, 1, 2], ln2)
        expected = [0.5, ln2/2, ln2**2/4]
        # 断言计算值与预期值接近
        assert_allclose(vals, expected)

    def test_mu0(self):
        # 边界情况：mu=0
        # 计算泊松分布的概率质量函数值
        vals = stats.poisson.pmf([0, 1, 2], 0)
        expected = [1, 0, 0]
        # 断言计算值与预期值相等
        assert_array_equal(vals, expected)
        # 计算 mu=0 时的置信区间
        interval = stats.poisson.interval(0.95, 0)
        assert_equal(interval, (0, 0))

    def test_rvs(self):
        # 随机变量生成测试
        # 生成泊松分布的随机变量
        vals = stats.poisson.rvs(0.5, size=(2, 50))
        # 断言所有随机变量均大于等于 0
        assert np.all(vals >= 0)
        # 断言生成的随机变量形状符合预期
        assert np.shape(vals) == (2, 50)
        # 断言生成的随机变量类型为整数
        assert vals.dtype.char in typecodes['AllInteger']
        # 生成单个泊松分布随机变量，断言其类型为整数
        val = stats.poisson.rvs(0.5)
        assert isinstance(val, int)
        # 生成多个泊松分布随机变量，断言其类型为 numpy 数组
        val = stats.poisson(0.5).rvs(3)
        assert isinstance(val, np.ndarray)
        # 断言生成的随机变量类型为整数
        assert val.dtype.char in typecodes['AllInteger']
    # 定义一个测试方法，用于验证泊松分布的统计功能
    def test_stats(self):
        # 设置泊松分布的参数 mu 为 16.0
        mu = 16.0
        # 调用 scipy.stats.poisson.stats 方法计算泊松分布的期望值、方差、偏度和峰度
        result = stats.poisson.stats(mu, moments='mvsk')
        # 使用 numpy.testing.assert_allclose 方法断言计算结果与期望值的接近程度
        assert_allclose(result, [mu, mu, np.sqrt(1.0/mu), 1.0/mu])

        # 将 mu 参数改为一个 numpy 数组 [0.0, 1.0, 2.0]
        mu = np.array([0.0, 1.0, 2.0])
        # 再次调用 scipy.stats.poisson.stats 方法计算多个泊松分布参数下的统计量
        result = stats.poisson.stats(mu, moments='mvsk')
        # 准备预期的结果，分别是期望值、方差、偏度和峰度
        expected = (mu, mu, [np.inf, 1, 1/np.sqrt(2)], [np.inf, 1, 0.5])
        # 断言计算结果与预期结果的接近程度
        assert_allclose(result, expected)
class TestKSTwo:
    # 在每个测试方法运行前设置随机种子为1234，以确保结果可复现性
    def setup_method(self):
        np.random.seed(1234)

    # 测试 Kolmogorov-Smirnov 分布的累积分布函数（CDF）
    def test_cdf(self):
        for n in [1, 2, 3, 10, 100, 1000]:
            # 测试的 x 值:
            #  0, 1/2n，此处 CDF 应为 0
            #  1/n，此处 CDF 应为 n!/n^n
            #  0.5，此处 CDF 应与 ksone.cdf 相匹配
            # 1-1/n，此处 CDF 应为 1-2/n^n
            # 1，此处 CDF 应为 1
            # （例如，由 Simard / L'Ecuyer 的方程1给出的确切值）
            x = np.array([0, 0.5/n, 1/n, 0.5, 1-1.0/n, 1])
            v1 = (1.0/n)**n
            lg = scipy.special.gammaln(n+1)
            elg = (np.exp(lg) if v1 != 0 else 0)
            expected = np.array([0, 0, v1 * elg,
                                 1 - 2*stats.ksone.sf(0.5, n),
                                 max(1 - 2*v1, 0.0),
                                 1.0])
            # 计算实际得到的 CDF 值
            vals_cdf = stats.kstwo.cdf(x, n)
            # 使用 assert_allclose 来比较计算得到的 CDF 值与预期的 CDF 值
            assert_allclose(vals_cdf, expected)

    # 测试 Kolmogorov-Smirnov 分布的生存函数（SF），即 1 - CDF
    def test_sf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            # 和 test_cdf 中相同的 x 值，并且使用 sf = 1 - cdf
            x = np.array([0, 0.5/n, 1/n, 0.5, 1-1.0/n, 1])
            v1 = (1.0/n)**n
            lg = scipy.special.gammaln(n+1)
            elg = (np.exp(lg) if v1 != 0 else 0)
            expected = np.array([1.0, 1.0,
                                 1 - v1 * elg,
                                 2*stats.ksone.sf(0.5, n),
                                 min(2*v1, 1.0), 0])
            # 计算实际得到的 SF 值
            vals_sf = stats.kstwo.sf(x, n)
            # 使用 assert_allclose 来比较计算得到的 SF 值与预期的 SF 值
            assert_allclose(vals_sf, expected)

    # 测试在 n 趋近无穷大时 Kolmogorov-Smirnov 分布的 CDF
    def test_cdf_sqrtn(self):
        # 对于固定的 a，当 n 趋近无穷大时，cdf(a/sqrt(n), n) -> kstwobign(a)
        # cdf(a/sqrt(n), n) 是 n（和 a）的递增函数
        # 检查该函数确实是递增的（允许一些小的浮点数和算法差异）
        x = np.linspace(0, 2, 11)[1:]
        ns = [50, 100, 200, 400, 1000, 2000]
        for _x in x:
            xn = _x / np.sqrt(ns)
            probs = stats.kstwo.cdf(xn, ns)
            diffs = np.diff(probs)
            # 使用 assert_array_less 检查 diffs 是否小于给定的阈值，以确保递增性
            assert_array_less(diffs, 1e-8)

    # 测试 CDF 和 SF 的关系，即 CDF(x, n) + SF(x, n) = 1
    def test_cdf_sf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            # 计算实际得到的 CDF 和 SF 值
            vals_cdf = stats.kstwo.cdf(x, n)
            vals_sf = stats.kstwo.sf(x, n)
            # 使用 assert_array_almost_equal 检查 CDF + SF 是否等于 1
            assert_array_almost_equal(vals_cdf, 1 - vals_sf)

    # 测试在 n 趋近无穷大时，通过调整 x 为 x / sqrt(n)，CDF 和 SF 的关系是否仍然成立
    def test_cdf_sf_sqrtn(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = x / np.sqrt(n)
            # 计算实际得到的 CDF 和 SF 值
            vals_cdf = stats.kstwo.cdf(xn, n)
            vals_sf = stats.kstwo.sf(xn, n)
            # 使用 assert_array_almost_equal 检查 CDF + SF 是否等于 1
            assert_array_almost_equal(vals_cdf, 1 - vals_sf)
    def test_ppf_of_cdf(self):
        # 生成一个从0到1的等间距数组，共11个元素，作为x值
        x = np.linspace(0, 1, 11)
        # 对于不同的n值进行循环测试
        for n in [1, 2, 3, 10, 100, 1000]:
            # 选择大于0.5/n的部分作为xn
            xn = x[x > 0.5/n]
            # 使用stats.kstwo.cdf计算xn对应的累积分布函数值
            vals_cdf = stats.kstwo.cdf(xn, n)
            # 对于CDF接近1的情况，更适合使用SF（Survival Function）
            cond = (0 < vals_cdf) & (vals_cdf < 0.99)
            # 使用stats.kstwo.ppf根据CDF值计算其对应的百分点函数值
            vals = stats.kstwo.ppf(vals_cdf, n)
            # 使用assert_allclose断言验证两者在rtol误差范围内是否相等
            assert_allclose(vals[cond], xn[cond], rtol=1e-4)

    def test_isf_of_sf(self):
        # 生成一个从0到1的等间距数组，共11个元素，作为x值
        x = np.linspace(0, 1, 11)
        # 对于不同的n值进行循环测试
        for n in [1, 2, 3, 10, 100, 1000]:
            # 选择大于0.5/n的部分作为xn
            xn = x[x > 0.5/n]
            # 使用stats.kstwo.isf计算xn对应的逆生存函数值
            vals_isf = stats.kstwo.isf(xn, n)
            # 对于ISF接近1的情况，更适合使用CDF
            cond = (0 < vals_isf) & (vals_isf < 1.0)
            # 使用stats.kstwo.sf计算逆生存函数值对应的生存函数值
            vals = stats.kstwo.sf(vals_isf, n)
            # 使用assert_allclose断言验证两者在rtol误差范围内是否相等
            assert_allclose(vals[cond], xn[cond], rtol=1e-4)

    def test_ppf_of_cdf_sqrtn(self):
        # 生成一个从0到1的等间距数组，共11个元素，作为x值
        x = np.linspace(0, 1, 11)
        # 对于不同的n值进行循环测试
        for n in [1, 2, 3, 10, 100, 1000]:
            # 计算x除以sqrt(n)后的值，并选择大于0.5/n的部分作为xn
            xn = (x / np.sqrt(n))[x > 0.5/n]
            # 使用stats.kstwo.cdf计算xn对应的累积分布函数值
            vals_cdf = stats.kstwo.cdf(xn, n)
            # 对于CDF接近1的情况，更适合使用1.0作为上限
            cond = (0 < vals_cdf) & (vals_cdf < 1.0)
            # 使用stats.kstwo.ppf根据CDF值计算其对应的百分点函数值
            vals = stats.kstwo.ppf(vals_cdf, n)
            # 使用assert_allclose断言验证两者是否相等
            assert_allclose(vals[cond], xn[cond])

    def test_isf_of_sf_sqrtn(self):
        # 生成一个从0到1的等间距数组，共11个元素，作为x值
        x = np.linspace(0, 1, 11)
        # 对于不同的n值进行循环测试
        for n in [1, 2, 3, 10, 100, 1000]:
            # 计算x除以sqrt(n)后的值，并选择大于0.5/n的部分作为xn
            xn = (x / np.sqrt(n))[x > 0.5/n]
            # 使用stats.kstwo.sf计算xn对应的生存函数值
            vals_sf = stats.kstwo.sf(xn, n)
            # 对于SF接近1的情况，更适合使用CDF
            cond = (0 < vals_sf) & (vals_sf < 0.95)
            # 使用stats.kstwo.isf计算生存函数值对应的逆生存函数值
            vals = stats.kstwo.isf(vals_sf, n)
            # 使用assert_allclose断言验证两者是否相等
            assert_allclose(vals[cond], xn[cond])

    def test_ppf(self):
        # 生成一个从0到1的等间距数组，共11个元素，取第2个到最后一个作为probs
        probs = np.linspace(0, 1, 11)[1:]
        # 对于不同的n值进行循环测试
        for n in [1, 2, 3, 10, 100, 1000]:
            # 使用stats.kstwo.ppf根据probs和n计算百分点函数值
            xn = stats.kstwo.ppf(probs, n)
            # 使用stats.kstwo.cdf计算ppf值对应的累积分布函数值
            vals_cdf = stats.kstwo.cdf(xn, n)
            # 使用assert_allclose断言验证两者是否相等
            assert_allclose(vals_cdf, probs)
    def test_simard_lecuyer_table1(self):
        # 计算接近分布均值的值的累积分布函数（CDF）
        # 均值 u ~ log(2)*sqrt(pi/(2n))
        # 计算 x 在 [u/4, u/3, u/2, u, 2u, 3u] 范围内的值
        # 这是根据Simard, R., L'Ecuyer, P. (2011)中的Table 1计算的
        # "Computing the Two-Sided Kolmogorov-Smirnov Distribution"。
        # 注意，以下数值不是从已发表的表格中获取的，而是使用独立的SageMath实现
        # Durbin算法（结合Marsaglia/Tsang/Wang版本的指数运算和缩放），
        # 使用500位算术计算得出的。
        # 已发表表格中的一些数值相对误差大于1e-4。
        
        # 不同的 n 值列表
        ns = [10, 50, 100, 200, 500, 1000]
        # ratios 数组，用于计算 x 的不同倍率
        ratios = np.array([1.0/4, 1.0/3, 1.0/2, 1, 2, 3])
        # 预期的累积分布函数值数组
        expected = np.array([
            [1.92155292e-08, 5.72933228e-05, 2.15233226e-02, 6.31566589e-01,
             9.97685592e-01, 9.99999942e-01],
            [2.28096224e-09, 1.99142563e-05, 1.42617934e-02, 5.95345542e-01,
             9.96177701e-01, 9.99998662e-01],
            [1.00201886e-09, 1.32673079e-05, 1.24608594e-02, 5.86163220e-01,
             9.95866877e-01, 9.99998240e-01],
            [4.93313022e-10, 9.52658029e-06, 1.12123138e-02, 5.79486872e-01,
             9.95661824e-01, 9.99997964e-01],
            [2.37049293e-10, 6.85002458e-06, 1.01309221e-02, 5.73427224e-01,
             9.95491207e-01, 9.99997750e-01],
            [1.56990874e-10, 5.71738276e-06, 9.59725430e-03, 5.70322692e-01,
             9.95409545e-01, 9.99997657e-01]
        ])
        
        # 对于每个 n 值，计算 x，并计算其累积分布函数值
        for idx, n in enumerate(ns):
            x = ratios * np.log(2) * np.sqrt(np.pi/2/n)
            vals_cdf = stats.kstwo.cdf(x, n)
            # 使用相对误差 rtol=1e-5 断言累积分布函数值与预期值接近
            assert_allclose(vals_cdf, expected[idx], rtol=1e-5)
class TestZipf:
    # 设置每个测试方法的初始状态
    def setup_method(self):
        np.random.seed(1234)

    # 测试 Zipf 分布的随机变量生成
    def test_rvs(self):
        # 生成参数为 1.5 的 Zipf 分布随机变量矩阵
        vals = stats.zipf.rvs(1.5, size=(2, 50))
        # 断言所有生成的随机变量都大于等于 1
        assert np.all(vals >= 1)
        # 断言生成的矩阵形状为 (2, 50)
        assert np.shape(vals) == (2, 50)
        # 断言生成的随机变量数据类型为整数
        assert vals.dtype.char in typecodes['AllInteger']
        # 生成参数为 1.5 的单个 Zipf 分布随机变量
        val = stats.zipf.rvs(1.5)
        # 断言生成的随机变量是整数
        assert isinstance(val, int)
        # 使用指定参数的 Zipf 分布对象生成三个随机变量
        val = stats.zipf(1.5).rvs(3)
        # 断言生成的随机变量数组是 NumPy 数组
        assert isinstance(val, np.ndarray)
        # 断言生成的随机变量数组数据类型为整数
        assert val.dtype.char in typecodes['AllInteger']

    # 测试 Zipf 分布的矩时刻
    def test_moments(self):
        # 计算参数为 2.8 的 Zipf 分布的均值和方差
        m, v = stats.zipf.stats(a=2.8)
        # 断言均值 m 是有限的
        assert_(np.isfinite(m))
        # 断言方差 v 为无穷大
        assert_equal(v, np.inf)

        # 计算参数为 4.8 的 Zipf 分布的斜度和峰度
        s, k = stats.zipf.stats(a=4.8, moments='sk')
        # 断言斜度 s 和峰度 k 都不是有限的
        assert_(not np.isfinite([s, k]).all())


class TestDLaplace:
    # 设置每个测试方法的初始状态
    def setup_method(self):
        np.random.seed(1234)

    # 测试双拉普拉斯分布的随机变量生成
    def test_rvs(self):
        # 生成参数为 1.5 的双拉普拉斯分布随机变量矩阵
        vals = stats.dlaplace.rvs(1.5, size=(2, 50))
        # 断言生成的矩阵形状为 (2, 50)
        assert np.shape(vals) == (2, 50)
        # 断言生成的随机变量数据类型为整数
        assert vals.dtype.char in typecodes['AllInteger']
        # 生成参数为 1.5 的单个双拉普拉斯分布随机变量
        val = stats.dlaplace.rvs(1.5)
        # 断言生成的随机变量是整数
        assert isinstance(val, int)
        # 使用指定参数的双拉普拉斯分布对象生成三个随机变量
        val = stats.dlaplace(1.5).rvs(3)
        # 断言生成的随机变量数组是 NumPy 数组
        assert isinstance(val, np.ndarray)
        # 断言生成的随机变量数组数据类型为整数
        assert val.dtype.char in typecodes['AllInteger']
        # 断言生成参数为 0.8 的双拉普拉斯分布随机变量不为 None
        assert stats.dlaplace.rvs(0.8) is not None

    # 测试双拉普拉斯分布的统计量
    def test_stats(self):
        # 使用显式公式与使用 pmf 进行直接求和比较
        a = 1.
        dl = stats.dlaplace(a)
        # 计算双拉普拉斯分布的均值、方差、斜度和峰度
        m, v, s, k = dl.stats('mvsk')

        N = 37
        xx = np.arange(-N, N+1)
        pp = dl.pmf(xx)
        m2, m4 = np.sum(pp*xx**2), np.sum(pp*xx**4)
        # 断言均值 m 和斜度 s 都为 0
        assert_equal((m, s), (0, 0))
        # 断言方差 v 和峰度 k 满足近似精度
        assert_allclose((v, k), (m2, m4/m2**2 - 3.), atol=1e-14, rtol=1e-8)

    # 测试双拉普拉斯分布的统计量（第二组）
    def test_stats2(self):
        a = np.log(2.)
        dl = stats.dlaplace(a)
        # 计算双拉普拉斯分布的均值、方差、斜度和峰度
        m, v, s, k = dl.stats('mvsk')
        # 断言均值 m 和斜度 s 都为 0
        assert_equal((m, s), (0., 0.))
        # 断言方差 v 和峰度 k 满足近似精度
        assert_allclose((v, k), (4., 3.25))


class TestInvgauss:
    # 设置每个测试方法的初始状态
    def setup_method(self):
        np.random.seed(1234)

    # 参数化测试：双拉普拉斯分布的随机变量生成
    @pytest.mark.parametrize("rvs_mu,rvs_loc,rvs_scale",
                             [(2, 0, 1), (4.635, 4.362, 6.303)])
    # 定义一个测试方法，用于测试拟合函数对于不同参数的行为
    def test_fit(self, rvs_mu, rvs_loc, rvs_scale):
        # 生成服从逆高斯分布的随机数据，其中 mu、loc、scale 是参数
        data = stats.invgauss.rvs(size=100, mu=rvs_mu, loc=rvs_loc, scale=rvs_scale)
        
        # 使用拟合函数计算分布的最大似然估计值（MLE），当 floc 参数固定时使用公式进行计算
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc)
        
        # 对数据进行调整，减去 rvs_loc
        data = data - rvs_loc
        # 计算临时的均值 mu_temp
        mu_temp = np.mean(data)
        # 计算基于 MLE 的 scale 估计值
        scale_mle = len(data) / (np.sum(data**(-1) - mu_temp**(-1)))
        # 计算基于 MLE 的 mu 估计值
        mu_mle = mu_temp / scale_mle
        
        # 断言 mu 和 scale 与解析公式匹配
        assert_allclose(mu_mle, mu, atol=1e-15, rtol=1e-15)
        assert_allclose(scale_mle, scale, atol=1e-15, rtol=1e-15)
        # 断言 loc 与 rvs_loc 相等
        assert_equal(loc, rvs_loc)
        
        # 重新生成服从逆高斯分布的随机数据，其中 mu、loc、scale 是参数
        data = stats.invgauss.rvs(size=100, mu=rvs_mu, loc=rvs_loc, scale=rvs_scale)
        
        # 返回固定参数的拟合结果
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc - 1, fscale=rvs_scale + 1)
        # 断言 scale 等于 rvs_scale + 1
        assert_equal(rvs_scale + 1, scale)
        # 断言 loc 等于 rvs_loc - 1
        assert_equal(rvs_loc - 1, loc)
        
        # 通过多个方式固定形状参数 mu = 1.04
        shape_mle1 = stats.invgauss.fit(data, fmu=1.04)[0]
        shape_mle2 = stats.invgauss.fit(data, fix_mu=1.04)[0]
        shape_mle3 = stats.invgauss.fit(data, f0=1.04)[0]
        # 断言三种方式得到的形状参数相等且等于 1.04
        assert shape_mle1 == shape_mle2 == shape_mle3 == 1.04

    # 使用 pytest 的参数化功能，为测试方法 test_fit 提供不同的参数组合进行测试
    @pytest.mark.parametrize("rvs_mu,rvs_loc,rvs_scale",
                             [(2, 0, 1), (6.311, 3.225, 4.520)])
    # 定义测试函数，用于测试极大似然估计（MLE）与不同优化器的适配性
    def test_fit_MLE_comp_optimizer(self, rvs_mu, rvs_loc, rvs_scale):
        # 设置随机数生成器的种子，以便复现随机结果
        rng = np.random.RandomState(1234)
        # 生成服从逆高斯分布的随机数据
        data = stats.invgauss.rvs(size=100, mu=rvs_mu,
                                  loc=rvs_loc, scale=rvs_scale, random_state=rng)

        # 获取父类（超类）的 fit 方法
        super_fit = super(type(stats.invgauss), stats.invgauss).fit
        # 使用超类的 fit 方法进行拟合，不使用 `floc` 参数
        super_fitted = super_fit(data)
        # 使用 stats.invgauss 的 fit 方法进行拟合
        invgauss_fit = stats.invgauss.fit(data)
        # 断言两种拟合结果相等
        assert_equal(super_fitted, invgauss_fit)

        # 使用超类的 fit 方法进行拟合，使用 `floc` 和 `fmu` 参数
        super_fitted = super_fit(data, floc=0, fmu=2)
        # 使用 stats.invgauss 的 fit 方法进行拟合，使用 `floc` 和 `fmu` 参数
        invgauss_fit = stats.invgauss.fit(data, floc=0, fmu=2)
        # 断言两种拟合结果相等
        assert_equal(super_fitted, invgauss_fit)

        # 使用固定的 `floc`，使用解析公式进行计算，并比超类方法提供更好的拟合
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc)

        # 使用固定的 `floc`，确保数据不会小于零，并使用解析公式比超类方法提供更好的拟合
        assert np.all((data - (rvs_loc - 1)) > 0)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc - 1)

        # 将 `floc` 固定为任意数值（这里为 0），仍然比超类方法提供更好的拟合
        _assert_less_or_close_loglike(stats.invgauss, data, floc=0)

        # 将 `fscale` 固定为任意数值，仍然比超类方法提供更好的拟合
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc,
                                      fscale=np.random.rand(1)[0])

    # 定义测试函数，用于验证拟合时是否会引发错误
    def test_fit_raise_errors(self):
        # 确保拟合时会发出警告
        assert_fit_warnings(stats.invgauss)
        # 当数据中存在小于零的无效数据时，应该引发 FitDataError 错误
        with pytest.raises(FitDataError):
            stats.invgauss.fit([1, 2, 3], floc=2)
    def test_cdf_sf(self):
        # 回归测试，用于检查 gh-13614 的修复效果。
        # 使用 R 的 statmod 库（pinvgauss）作为基准。
        # 示例代码：
        # library(statmod)
        # options(digits=15)
        # mu = c(4.17022005e-04, 7.20324493e-03, 1.14374817e-06,
        #        3.02332573e-03, 1.46755891e-03)
        # print(pinvgauss(5, mu, 1))

        # 确保当 mu 非常小时返回有限值。参见 GH-13614
        mu = [4.17022005e-04, 7.20324493e-03, 1.14374817e-06,
              3.02332573e-03, 1.46755891e-03]
        expected = [1, 1, 1, 1, 1]
        actual = stats.invgauss.cdf(0.4, mu=mu)
        assert_equal(expected, actual)

        # 测试函数能否区分左/右小尾概率不为零的情况。
        cdf_actual = stats.invgauss.cdf(0.001, mu=1.05)
        assert_allclose(cdf_actual, 4.65246506892667e-219)
        sf_actual = stats.invgauss.sf(110, mu=1.05)
        assert_allclose(sf_actual, 4.12851625944048e-25)

        # 当 mu 非常小时，确保 x 不会引起数值问题，尤其当 x 接近 mu 时。
        
        # 稍微小于 mu 的情况
        actual = stats.invgauss.cdf(0.00009, 0.0001)
        assert_allclose(actual, 2.9458022894924e-26)

        # 稍微大于 mu 的情况
        actual = stats.invgauss.cdf(0.000102, 0.0001)
        assert_allclose(actual, 0.976445540507925)

    def test_logcdf_logsf(self):
        # 回归测试，用于检查 gh-13616 的改进效果。
        # 使用 R 的 statmod 库（pinvgauss）作为基准。
        # 示例代码：
        # library(statmod)
        # options(digits=15)
        # print(pinvgauss(0.001, 1.05, 1, log.p=TRUE, lower.tail=FALSE))

        # 测试 logcdf 和 logsf 能否计算小到无法在未取对数的尺度上表示的值。参见：gh-13616
        logcdf = stats.invgauss.logcdf(0.0001, mu=1.05)
        assert_allclose(logcdf, -5003.87872590367)
        logcdf = stats.invgauss.logcdf(110, 1.05)
        assert_allclose(logcdf, -4.12851625944087e-25)
        logsf = stats.invgauss.logsf(0.001, mu=1.05)
        assert_allclose(logsf, -4.65246506892676e-219)
        logsf = stats.invgauss.logsf(110, 1.05)
        assert_allclose(logsf, -56.1467092416426)

    # from mpmath import mp
    # mp.dps = 100
    # mu = mp.mpf(1e-2)
    # ref = (1/2 * mp.log(2 * mp.pi * mp.e * mu**3)
    #        - 3/2* mp.exp(2/mu) * mp.e1(2/mu))
    @pytest.mark.parametrize("mu, ref", [(2e-8, -25.172361826883957),
                                         (1e-3, -8.943444010642972),
                                         (1e-2, -5.4962796152622335),
                                         (1e8, 3.3244822568873476),
                                         (1e100, 3.32448280139689)])
    def test_entropy(self, mu, ref):
        # 测试信息熵计算函数的准确性。
        assert_allclose(stats.invgauss.entropy(mu), ref, rtol=5e-14)
# 定义一个测试类 TestLaplace，用于测试拉普拉斯分布相关的功能
class TestLaplace:
    # 使用 pytest 的参数化装饰器，分别对 rvs_loc 和 rvs_scale 参数进行参数化
    @pytest.mark.parametrize("rvs_loc", [-5, 0, 1, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 2, 3, 10])
    # 定义测试方法 test_fit，用于测试拟合方法的正确性
    def test_fit(self, rvs_loc, rvs_scale):
        # 测试各种输入是否符合预期行为，基于不同的 loc 和 scale 参数
        rng = np.random.RandomState(1234)  # 创建一个指定种子的随机数生成器
        # 生成服从拉普拉斯分布的随机数据，指定 loc 和 scale 参数
        data = stats.laplace.rvs(size=100, loc=rvs_loc, scale=rvs_scale,
                                 random_state=rng)

        # 计算数据的最大似然估计值
        loc_mle = np.median(data)  # 使用中位数作为 loc 的估计值
        scale_mle = np.sum(np.abs(data - loc_mle)) / len(data)  # 计算 scale 的估计值

        # 检查标准输出是否与解析的最大似然估计公式匹配
        loc, scale = stats.laplace.fit(data)  # 使用 stats.laplace.fit 进行拟合
        assert_allclose(loc, loc_mle, atol=1e-15, rtol=1e-15)  # 检查 loc 是否接近 loc_mle
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)  # 检查 scale 是否接近 scale_mle

        # 使用指定的 loc_mle 进行固定参数拟合，检查 scale 是否匹配
        loc, scale = stats.laplace.fit(data, floc=loc_mle)
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)
        # 使用指定的 scale_mle 进行固定参数拟合，检查 loc 是否匹配
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_allclose(loc, loc_mle)

        # 使用非中位数的 loc 进行固定参数拟合，检查 scale 是否匹配
        loc = rvs_loc * 2
        scale_mle = np.sum(np.abs(data - loc)) / len(data)
        loc, scale = stats.laplace.fit(data, floc=loc)
        assert_equal(scale_mle, scale)

        # 使用非中位数 loc 创建的 scale，检查 loc 输出是否仍为数据的中位数
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_equal(loc_mle, loc)

        # 当同时固定 floc 和 fscale 时，应该引发 RuntimeError
        assert_raises(RuntimeError, stats.laplace.fit, data, floc=loc_mle,
                      fscale=scale_mle)

        # 当数据包含非有限值时，应该引发 ValueError
        assert_raises(ValueError, stats.laplace.fit, [np.nan])
        assert_raises(ValueError, stats.laplace.fit, [np.inf])

    # 使用参数化装饰器对 rvs_loc 和 rvs_scale 参数进行进一步的参数化
    @pytest.mark.parametrize("rvs_loc,rvs_scale", [(-5, 10),
                                                   (10, 5),
                                                   (0.5, 0.2)])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale):
        # 使用种子为1234的随机数生成器创建随机数生成器对象
        rng = np.random.RandomState(1234)
        # 从拉普拉斯分布生成大小为1000的随机样本数据
        data = stats.laplace.rvs(size=1000, loc=rvs_loc, scale=rvs_scale,
                                 random_state=rng)

        # 拉普拉斯分布的对数似然函数定义为
        def ll(loc, scale, data):
            return -1 * (- (len(data)) * np.log(2*scale) -
                         (1/scale)*np.sum(np.abs(data - loc)))

        # 测试解析最大似然估计的目标函数结果是否小于或等于数值优化估计的结果
        # 使用样本数据计算最大似然估计值
        loc, scale = stats.laplace.fit(data)
        # 使用超类方法计算数值优化估计值
        loc_opt, scale_opt = super(type(stats.laplace),
                                   stats.laplace).fit(data)
        # 计算最大似然估计值的对数似然函数值和数值优化估计值的对数似然函数值
        ll_mle = ll(loc, scale, data)
        ll_opt = ll(loc_opt, scale_opt, data)
        # 断言最大似然估计值的对数似然函数值小于数值优化估计值的对数似然函数值，
        # 或者它们在给定精度下非常接近
        assert ll_mle < ll_opt or np.allclose(ll_mle, ll_opt,
                                              atol=1e-15, rtol=1e-15)

    def test_fit_simple_non_random_data(self):
        # 定义一个简单的非随机数据数组
        data = np.array([1.0, 1.0, 3.0, 5.0, 8.0, 14.0])
        # 当`floc`固定为6时，应该得到的scale值是4
        loc, scale = stats.laplace.fit(data, floc=6)
        # 断言得到的scale值接近于4
        assert_allclose(scale, 4, atol=1e-15, rtol=1e-15)
        # 当`fscale`固定为6时，应该得到的loc值是4
        loc, scale = stats.laplace.fit(data, fscale=6)
        # 断言得到的loc值接近于4
        assert_allclose(loc, 4, atol=1e-15, rtol=1e-15)

    def test_sf_cdf_extremes(self):
        # 这些计算不应产生警告
        x = 1000
        # 计算拉普拉斯分布的累积分布函数在-x处的值
        p0 = stats.laplace.cdf(-x)
        # 由于精度限制，期望的结果是0
        assert p0 == 0.0
        # 最接近的64位浮点表示的确切值是1.0
        p1 = stats.laplace.cdf(x)
        assert p1 == 1.0

        # 计算拉普拉斯分布的生存函数在x处的值
        p0 = stats.laplace.sf(x)
        # 由于精度限制，期望的结果是0
        assert p0 == 0.0
        # 最接近的64位浮点表示的确切值是1.0
        p1 = stats.laplace.sf(-x)
        assert p1 == 1.0

    def test_sf(self):
        x = 200
        # 计算拉普拉斯分布的生存函数在x处的值
        p = stats.laplace.sf(x)
        # 断言计算得到的生存函数值与预期值非常接近
        assert_allclose(p, np.exp(-x)/2, rtol=1e-13)

    def test_isf(self):
        p = 1e-25
        # 计算拉普拉斯分布的逆生存函数在概率p处的值
        x = stats.laplace.isf(p)
        # 断言计算得到的逆生存函数值与预期值非常接近
        assert_allclose(x, -np.log(2*p), rtol=1e-13)
class TestLogLaplace:

    def test_sf(self):
        # 参考值是通过参考分布计算得出的，例如 mp.dps = 100; LogLaplace(c=c).sf(x).
        c = np.array([2.0, 3.0, 5.0])  # 设置参数 c
        x = np.array([1e-5, 1e10, 1e15])  # 设置参数 x
        ref = [0.99999999995, 5e-31, 5e-76]  # 参考值列表
        assert_allclose(stats.loglaplace.sf(x, c), ref, rtol=1e-15)  # 断言计算的 sf 值与参考值的接近程度

    def test_isf(self):
        # 参考值是通过参考分布计算得出的，例如 mp.dps = 100; LogLaplace(c=c).isf(q).
        c = 3.25  # 设置参数 c
        q = [0.8, 0.1, 1e-10, 1e-20, 1e-40]  # 设置参数 q
        ref = [0.7543222539245642, 1.6408455124660906, 964.4916294395846,
               1151387.578354072, 1640845512466.0906]  # 参考值列表
        assert_allclose(stats.loglaplace.isf(q, c), ref, rtol=1e-14)  # 断言计算的 isf 值与参考值的接近程度

    @pytest.mark.parametrize('r', [1, 2, 3, 4])
    def test_moments_stats(self, r):
        mom = 'mvsk'[r - 1]  # 根据 r 的值选择 'mvsk' 中的一个字符
        c = np.arange(0.5, r + 0.5, 0.5)  # 创建一个数组 c，范围从 0.5 到 r+0.5，步长为 0.5

        # 如果 |r| >= c，则 r-th 非中心矩是无限的。
        assert_allclose(stats.loglaplace.moment(r, c), np.inf)

        # 如果 r >= c，则 r-th 非中心矩是非有限的（无限或NaN）。
        assert not np.any(np.isfinite(stats.loglaplace.stats(c, moments=mom)))

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("loc, scale", [(-1.2, 3.45)])
    @pytest.mark.parametrize("fix_c", [True, False])
    @pytest.mark.parametrize("fix_scale", [True, False])
    def test_fit_analytic_mle(self, c, loc, scale, fix_c, fix_scale):
        # 测试分析MLE是否产生的结果不比通用（数值）MLE差。

        rng = np.random.default_rng(6762668991392531563)
        data = stats.loglaplace.rvs(c, loc=loc, scale=scale, size=100,
                                    random_state=rng)  # 生成 Log-Laplace 分布的随机样本

        kwds = {'floc': loc}  # 关键字参数字典，包含位置参数 loc
        if fix_c:
            kwds['fc'] = c  # 如果 fix_c 为 True，则添加参数 fc
        if fix_scale:
            kwds['fscale'] = scale  # 如果 fix_scale 为 True，则添加参数 fscale
        nfree = 3 - len(kwds)  # 自由参数的数量

        if nfree == 0:
            error_msg = "All parameters fixed. There is nothing to optimize."
            with pytest.raises((RuntimeError, ValueError), match=error_msg):
                stats.loglaplace.fit(data, **kwds)  # 如果所有参数都被固定，引发错误
            return

        _assert_less_or_close_loglike(stats.loglaplace, data, **kwds)


class TestPowerlaw:

    # 在以下数据中，`sf` 是用 mpmath 计算得出的。
    @pytest.mark.parametrize('x, a, sf',
                             [(0.25, 2.0, 0.9375),
                              (0.99609375, 1/256, 1.528855235208108e-05)])
    def test_sf(self, x, a, sf):
        assert_allclose(stats.powerlaw.sf(x, a), sf, rtol=1e-15)  # 断言计算的 sf 值与参考值的接近程度

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)  # 返回一个随机数生成器对象

    @pytest.mark.parametrize("rvs_shape", [.1, .5, .75, 1, 2])
    @pytest.mark.parametrize("rvs_loc", [-1, 0, 1])
    @pytest.mark.parametrize("rvs_scale", [.1, 1, 5])
    # 使用 pytest 的参数化装饰器，为测试方法提供多组参数组合
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [p for p in product([True, False], repeat=3)
                              if False in p])
    # 定义测试方法，用于测试最大似然估计与优化器组合的情况
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        # 生成符合幂律分布的随机数据
        data = stats.powerlaw.rvs(size=250, a=rvs_shape, loc=rvs_loc,
                                  scale=rvs_scale, random_state=rng)

        kwds = dict()
        # 根据参数设置需要固定的形状、位置和尺度
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            # 将位置参数设置为数据中最小值的下一个浮点数
            kwds['floc'] = np.nextafter(data.min(), -np.inf)
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 断言数值结果可以与解析结果相等，如果解析过程中的某些代码路径使用了数值优化
        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds,
                                      maybe_identical=True)

    # 定义一个测试特定问题情况的方法
    def test_problem_case(self):
        # 设定幂律分布的参数
        a = 2.50002862645130604506
        location = 0.0
        scale = 35.249023299873095

        # 生成符合指定参数的随机数据
        data = stats.powerlaw.rvs(a=a, loc=location, scale=scale, size=100,
                                  random_state=np.random.default_rng(5))

        kwds = {'fscale': np.ptp(data) * 2}

        # 断言数值结果可以小于或接近解析结果的对数似然值
        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds)

    # 定义测试最大似然估计过程中的警告情况的方法
    def test_fit_warnings(self):
        # 断言执行最大似然估计时会出现警告
        assert_fit_warnings(stats.powerlaw)

        # 当满足条件 `fscale + floc <= np.max(data)` 时，预期抛出 FitDataError 错误
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=0, fscale=3)

        # 当满足条件 `data - floc >= 0` 时，预期抛出 FitDataError 错误
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=2)

        # 当固定位置小于数据的最小值时，预期抛出 FitDataError 错误
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=1)

        # 当固定尺度小于或等于数据范围时，预期抛出 ValueError 错误
        msg = r"Negative or zero `fscale` is outside"
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=-3)

        # 当固定尺度小于数据范围时，预期抛出 ValueError 错误
        msg = r"`fscale` must be greater than the range of data."
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=3)
    def test_minimum_data_zero_gh17801(self):
        # 定义一个测试函数，用于验证 gh-17801 报告的在数据最小值为零时可能发生溢出错误的问题是否已解决。
        data = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6]
        # 使用 powerlaw 分布进行测试
        dist = stats.powerlaw
        # 在计算期间忽略 'over' 类型的 numpy 错误
        with np.errstate(over='ignore'):
            # 调用 _assert_less_or_close_loglike 函数，验证分布和数据的对数似然性
            _assert_less_or_close_loglike(dist, data)
class TestPowerLogNorm:

    # reference values were computed via mpmath

    # 定义了一个测试类 TestPowerLogNorm，用于测试 powerlognorm 模块的函数

    @pytest.mark.parametrize("x, c, s, ref",
                             [(100, 20, 1, 1.9057100820561928e-114),
                              (1e-3, 20, 1, 0.9999999999507617),
                              (1e-3, 0.02, 1, 0.9999999999999508),
                              (1e22, 0.02, 1, 6.50744044621611e-12)])
    def test_sf(self, x, c, s, ref):
        # 使用 pytest 的 parametrize 装饰器定义多组测试参数和预期结果
        assert_allclose(stats.powerlognorm.sf(x, c, s), ref, rtol=1e-13)
        # 调用被测试的 survival function，并验证其返回值与预期结果的接近度

    # reference values were computed via mpmath using the survival
    # function above (passing in `ref` and getting `q`).
    @pytest.mark.parametrize("q, c, s, ref",
                             [(0.9999999587870905, 0.02, 1, 0.01),
                              (6.690376686108851e-233, 20, 1, 1000)])
    def test_isf(self, q, c, s, ref):
        # 使用 pytest 的 parametrize 装饰器定义多组测试参数和预期结果
        assert_allclose(stats.powerlognorm.isf(q, c, s), ref, rtol=5e-11)
        # 调用被测试的 inverse survival function，并验证其返回值与预期结果的接近度

    @pytest.mark.parametrize("x, c, s, ref",
                             [(1e25, 0.02, 1, 0.9999999999999963),
                              (1e-6, 0.02, 1, 2.054921078040843e-45),
                              (1e-6, 200, 1, 2.0549210780408428e-41),
                              (0.3, 200, 1, 0.9999999999713368)])
    def test_cdf(self, x, c, s, ref):
        # 使用 pytest 的 parametrize 装饰器定义多组测试参数和预期结果
        assert_allclose(stats.powerlognorm.cdf(x, c, s), ref, rtol=3e-14)
        # 调用被测试的 cumulative distribution function，并验证其返回值与预期结果的接近度

    @pytest.mark.parametrize("x, c, s, ref",
                             [(1e22, 0.02, 1, 6.5954987852335016e-34),
                              (1e20, 1e-3, 1, 1.588073750563988e-22),
                              (1e40, 1e-3, 1, 1.3179391812506349e-43)])
    def test_pdf(self, x, c, s, ref):
        # 使用 pytest 的 parametrize 装饰器定义多组测试参数和预期结果
        assert_allclose(stats.powerlognorm.pdf(x, c, s), ref, rtol=3e-12)
        # 调用被测试的 probability density function，并验证其返回值与预期结果的接近度


class TestPowerNorm:

    # survival function references were computed with mpmath via

    # 定义了一个测试类 TestPowerNorm，用于测试 powernorm 模块的函数

    @pytest.mark.parametrize("x, c, ref",
                             [(9, 1, 1.1285884059538405e-19),
                              (20, 2, 7.582445786569958e-178),
                              (100, 0.02, 3.330957891903866e-44),
                              (200, 0.01, 1.3004759092324774e-87)])
    # 定义一个测试方法，用于检验幂次正态分布的生存函数 (sf) 的计算是否正确
    def test_sf(self, x, c, ref):
        # 使用 assert_allclose 断言函数检验 stats.powernorm.sf(x, c) 的计算结果是否接近于 ref
        assert_allclose(stats.powernorm.sf(x, c), ref, rtol=1e-13)

    # 下面是一些反函数生存函数的参考值，通过 mpmath 计算
    # 从 mpmath 库导入 mp
    # def isf_mp(q, c):
    #     q = mp.mpf(q)
    #     c = mp.mpf(c)
    #     arg = q**(mp.one / c)
    #     return float(-mp.sqrt(2) * mp.erfinv(mp.mpf(2.) * arg - mp.one))

    # 使用 pytest 的 parametrize 标记，对多组参数进行测试
    @pytest.mark.parametrize("q, c, ref",
                             [(1e-5, 20, -0.15690800666514138),
                              (0.99999, 100, -5.19933666203545),
                              (0.9999, 0.02, -2.576676052143387),
                              (5e-2, 0.02, 17.089518110222244),
                              (1e-18, 2, 5.9978070150076865),
                              (1e-50, 5, 6.361340902404057)])
    # 定义一个测试方法，用于检验幂次正态分布的反函数生存函数 (isf) 的计算是否正确
    def test_isf(self, q, c, ref):
        # 使用 assert_allclose 断言函数检验 stats.powernorm.isf(q, c) 的计算结果是否接近于 ref
        assert_allclose(stats.powernorm.isf(q, c), ref, rtol=5e-12)

    # 下面是一些累积分布函数 (CDF) 的参考值，通过 mpmath 计算
    # from mpmath import mp
    # def cdf_mp(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return float(mp.one - mp.ncdf(-x)**c)

    # 使用 pytest 的 parametrize 标记，对多组参数进行测试
    @pytest.mark.parametrize("x, c, ref",
                             [(-12, 9, 1.598833900869911e-32),
                              (2, 9, 0.9999999999999983),
                              (-20, 9, 2.4782617067456103e-88),
                              (-5, 0.02, 5.733032242841443e-09),
                              (-20, 0.02, 5.507248237212467e-91)])
    # 定义一个测试方法，用于检验幂次正态分布的累积分布函数 (cdf) 的计算是否正确
    def test_cdf(self, x, c, ref):
        # 使用 assert_allclose 断言函数检验 stats.powernorm.cdf(x, c) 的计算结果是否接近于 ref
        assert_allclose(stats.powernorm.cdf(x, c), ref, rtol=5e-14)
class TestInvGamma:
    def test_invgamma_inf_gh_1866(self):
        # invgamma's moments are only finite for a>n
        # specific numbers checked w/ boost 1.54
        # 使用警告模块捕获运行时警告
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # 计算逆伽马分布的统计量（m, v, s, k）
            mvsk = stats.invgamma.stats(a=19.31, moments='mvsk')
            expected = [0.05461496450, 0.0001723162534, 1.020362676,
                        2.055616582]
            # 断言计算结果与预期值接近
            assert_allclose(mvsk, expected)

            # 测试多个参数情况下的逆伽马分布统计量（m, v, s, k）
            a = [1.1, 3.1, 5.6]
            mvsk = stats.invgamma.stats(a=a, moments='mvsk')
            expected = ([10., 0.476190476, 0.2173913043],       # mmm
                        [np.inf, 0.2061430632, 0.01312749422],  # vvv
                        [np.nan, 41.95235392, 2.919025532],     # sss
                        [np.nan, np.nan, 24.51923076])          # kkk
            # 逐一断言每个统计量与预期值接近
            for x, y in zip(mvsk, expected):
                assert_almost_equal(x, y)

    def test_cdf_ppf(self):
        # gh-6245
        # 使用对数空间生成测试数据
        x = np.logspace(-2.6, 0)
        # 计算逆伽马分布的累积分布函数
        y = stats.invgamma.cdf(x, 1)
        # 使用累积分布函数的值计算逆伽马分布的百分点函数
        xx = stats.invgamma.ppf(y, 1)
        # 断言计算结果与输入数据接近
        assert_allclose(x, xx)

    def test_sf_isf(self):
        # gh-6245
        # 根据系统位数判断使用的测试数据范围
        if sys.maxsize > 2**32:
            x = np.logspace(2, 100)
        else:
            # 在32位系统上，逆伽马分布的往复计算具有相对精度约为1e-15
            # 直到 x=1e+15，而在 x=1e+18 以上则为无穷
            x = np.logspace(2, 18)

        # 计算逆伽马分布的生存函数
        y = stats.invgamma.sf(x, 1)
        # 使用生存函数的值计算逆伽马分布的逆生存函数
        xx = stats.invgamma.isf(y, 1)
        # 断言计算结果与输入数据接近，设置相对容差为1.0
        assert_allclose(x, xx, rtol=1.0)

    @pytest.mark.parametrize("a, ref",
                             [(100000000.0, -26.21208257605721),
                              (1e+100, -343.9688254159022)])
    def test_large_entropy(self, a, ref):
        # 参考值是使用 mpmath 计算得到的:
        # from mpmath import mp
        # mp.dps = 500

        # def invgamma_entropy(a):
        #     a = mp.mpf(a)
        #     h = a + mp.loggamma(a) - (mp.one + a) * mp.digamma(a)
        #     return float(h)
        # 断言逆伽马分布的熵与参考值接近，设置相对容差为1e-15
        assert_allclose(stats.invgamma.entropy(a), ref, rtol=1e-15)


class TestF:
    def test_endpoints(self):
        # 计算概率密度函数在左端点 dst.a 处的值
        data = [[stats.f, (2, 1), 1.0]]
        for _f, _args, _correct in data:
            ans = _f.pdf(_f.a, *_args)

        # 对数据集中的每个函数计算 pdf 值并与预期值比较
        ans = [_f.pdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [_correct_ for _f, _args, _correct_ in data]
        # 断言计算结果与预期值接近
        assert_array_almost_equal(ans, correct)

    def test_f_moments(self):
        # F 分布的 n-th 阶矩仅在 n < dfd / 2 时有限
        m, v, s, k = stats.f.stats(11, 6.5, moments='mvsk')
        # 断言平均值、方差、偏度和峰度都是有限的
        assert_(np.isfinite(m))
        assert_(np.isfinite(v))
        assert_(np.isfinite(s))
        assert_(not np.isfinite(k))
    # 定义一个测试方法，用于测试 moments 警告
    def test_moments_warnings(self):
        # 捕获警告，确保在特定条件下不会生成警告（即 dfd = 2, 4, 6, 8 时会出现除以零的情况）
        with warnings.catch_warnings():
            # 设置警告过滤器，将 RuntimeWarning 转换为异常
            warnings.simplefilter('error', RuntimeWarning)
            # 调用 stats.f.stats 函数，对于 dfn=[11]*4, dfd=[2, 4, 6, 8]，计算 'mvsk' 统计量
            stats.f.stats(dfn=[11]*4, dfd=[2, 4, 6, 8], moments='mvsk')

    # 定义一个测试方法，用于测试 stats 函数的广播功能
    def test_stats_broadcast(self):
        # 创建一个包含多维数组的 dfn 变量
        dfn = np.array([[3], [11]])
        # 创建一个包含多个元素的 dfd 变量
        dfd = np.array([11, 12])
        # 调用 stats.f.stats 函数，计算 'mvsk' 统计量，并将结果分别赋给 m, v, s, k 变量
        m, v, s, k = stats.f.stats(dfn=dfn, dfd=dfd, moments='mvsk')
        # 根据 dfd 的值计算 m2 变量，这里使用了广播操作
        m2 = [dfd / (dfd - 2)]*2
        # 使用 assert_allclose 断言函数，验证 m 和 m2 的近似性
        assert_allclose(m, m2)
        # 根据 dfn 和 dfd 的值计算 v2 变量，这里也使用了广播操作
        v2 = 2 * dfd**2 * (dfn + dfd - 2) / dfn / (dfd - 2)**2 / (dfd - 4)
        # 使用 assert_allclose 断言函数，验证 v 和 v2 的近似性
        assert_allclose(v, v2)
        # 根据 dfn 和 dfd 的值计算 s2 变量，这里包含了一些数学计算和广播操作
        s2 = ((2*dfn + dfd - 2) * np.sqrt(8*(dfd - 4)) /
              ((dfd - 6) * np.sqrt(dfn*(dfn + dfd - 2))))
        # 使用 assert_allclose 断言函数，验证 s 和 s2 的近似性
        assert_allclose(s, s2)
        # 根据 dfn 和 dfd 的值计算 k2num 和 k2den 变量，然后计算 k2 变量
        k2num = 12 * (dfn * (5*dfd - 22) * (dfn + dfd - 2) +
                      (dfd - 4) * (dfd - 2)**2)
        k2den = dfn * (dfd - 6) * (dfd - 8) * (dfn + dfd - 2)
        k2 = k2num / k2den
        # 使用 assert_allclose 断言函数，验证 k 和 k2 的近似性
        assert_allclose(k, k2)
class TestStudentT:
    # 测试标准偏差的回归测试，针对问题 #1191
    def test_rvgeneric_std(self):
        assert_array_almost_equal(stats.t.std([5, 6]), [1.29099445, 1.22474487])

    # 对 t 分布的矩计算进行回归测试，针对问题 #8786
    def test_moments_t(self):
        assert_equal(stats.t.stats(df=1, moments='mvsk'),
                     (np.inf, np.nan, np.nan, np.nan))
        assert_equal(stats.t.stats(df=1.01, moments='mvsk'),
                     (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2, moments='mvsk'),
                     (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2.01, moments='mvsk'),
                     (0.0, 2.01/(2.01-2.0), np.nan, np.inf))
        assert_equal(stats.t.stats(df=3, moments='sk'), (np.nan, np.inf))
        assert_equal(stats.t.stats(df=3.01, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4.01, moments='sk'), (0.0, 6.0/(4.01 - 4.0)))

    # 对 t 分布的熵计算进行测试
    def test_t_entropy(self):
        df = [1, 2, 25, 100]
        # 期望值是使用 mpmath 计算得出的
        expected = [2.5310242469692907, 1.9602792291600821,
                    1.459327578078393, 1.4289633653182439]
        assert_allclose(stats.t.entropy(df), expected, rtol=1e-13)

    # 对 t 分布的极端熵计算进行测试
    @pytest.mark.parametrize("v, ref",
                             [(100, 1.4289633653182439),
                              (1e+100, 1.4189385332046727)])
    def test_t_extreme_entropy(self, v, ref):
        # 参考值是使用 mpmath 计算得出的:
        # from mpmath import mp
        # mp.dps = 500
        #
        # def t_entropy(v):
        #   v = mp.mpf(v)
        #   C = (v + mp.one) / 2
        #   A = C * (mp.digamma(C) - mp.digamma(v / 2))
        #   B = 0.5 * mp.log(v) + mp.log(mp.beta(v / 2, mp.one / 2))
        #   h = A + B
        #   return float(h)
        assert_allclose(stats.t.entropy(v), ref, rtol=1e-14)

    # 对 t 分布的不同方法进行参数化测试
    @pytest.mark.parametrize("methname", ["pdf", "logpdf", "cdf",
                                          "ppf", "sf", "isf"])
    @pytest.mark.parametrize("df_infmask", [[0, 0], [1, 1], [0, 1],
                                            [[0, 1, 0], [1, 1, 1]],
                                            [[1, 0], [0, 1]],
                                            [[0], [1]]])
    def test_t_inf_df(self, methname, df_infmask):
        # 设置随机数种子为0，以确保结果可复现
        np.random.seed(0)
        # 将 df_infmask 转换为布尔类型的 NumPy 数组
        df_infmask = np.asarray(df_infmask, dtype=bool)
        # 生成一个与 df_infmask 形状相同的均匀分布的随机数组
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        # 生成一个与 df_infmask 形状相同的标准正态分布的随机数组
        x = np.random.randn(*df_infmask.shape)
        # 将 df 中 df_infmask 对应的位置设为正无穷
        df[df_infmask] = np.inf
        # 创建 t 分布的对象，指定自由度为 df，位置参数为 3，尺度参数为 1
        t_dist = stats.t(df=df, loc=3, scale=1)
        # 创建 t 分布的参考对象，用于无穷值位置的替代
        t_dist_ref = stats.t(df=df[~df_infmask], loc=3, scale=1)
        # 创建标准正态分布的对象，位置参数为 3，尺度参数为 1
        norm_dist = stats.norm(loc=3, scale=1)
        # 获取 t_dist 对象的指定方法的可调用对象
        t_meth = getattr(t_dist, methname)
        # 获取 t_dist_ref 对象的指定方法的可调用对象
        t_meth_ref = getattr(t_dist_ref, methname)
        # 获取 norm_dist 对象的指定方法的可调用对象
        norm_meth = getattr(norm_dist, methname)
        # 计算 t_meth 对象在输入 x 上的结果
        res = t_meth(x)
        # 断言 t_meth 对象在 df_infmask 为 True 的位置上与 norm_meth 在相同位置的结果相等
        assert_equal(res[df_infmask], norm_meth(x[df_infmask]))
        # 断言 t_meth 对象在 df_infmask 为 False 的位置上与 t_meth_ref 在相同位置的结果相等
        assert_equal(res[~df_infmask], t_meth_ref(x[~df_infmask]))

    @pytest.mark.parametrize("df_infmask", [[0, 0], [1, 1], [0, 1],
                                            [[0, 1, 0], [1, 1, 1]],
                                            [[1, 0], [0, 1]],
                                            [[0], [1]]])
    def test_t_inf_df_stats_entropy(self, df_infmask):
        # 设置随机数种子为0，以确保结果可复现
        np.random.seed(0)
        # 将 df_infmask 转换为布尔类型的 NumPy 数组
        df_infmask = np.asarray(df_infmask, dtype=bool)
        # 生成一个与 df_infmask 形状相同的均匀分布的随机数组
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        # 将 df 中 df_infmask 对应的位置设为正无穷
        df[df_infmask] = np.inf
        # 计算 t 分布的统计量，包括均值、方差、偏度和峰度
        res = stats.t.stats(df=df, loc=3, scale=1, moments='mvsk')
        # 计算标准正态分布的统计量，包括均值、方差、偏度和峰度
        res_ex_inf = stats.norm.stats(loc=3, scale=1, moments='mvsk')
        # 计算 t 分布在去除正无穷值后的统计量，包括均值、方差、偏度和峰度
        res_ex_noinf = stats.t.stats(df=df[~df_infmask], loc=3, scale=1,
                                    moments='mvsk')
        # 遍历每个统计量，断言在 df_infmask 为 True 的位置上与 res_ex_inf 相等
        for i in range(4):
            assert_equal(res[i][df_infmask], res_ex_inf[i])
            # 断言在 df_infmask 为 False 的位置上与 res_ex_noinf 相等
            assert_equal(res[i][~df_infmask], res_ex_noinf[i])

        # 计算 t 分布的熵
        res = stats.t.entropy(df=df, loc=3, scale=1)
        # 计算标准正态分布的熵
        res_ex_inf = stats.norm.entropy(loc=3, scale=1)
        # 计算在去除正无穷值后的 t 分布的熵
        res_ex_noinf = stats.t.entropy(df=df[~df_infmask], loc=3, scale=1)
        # 断言在 df_infmask 为 True 的位置上与 res_ex_inf 相等
        assert_equal(res[df_infmask], res_ex_inf)
        # 断言在 df_infmask 为 False 的位置上与 res_ex_noinf 相等
        assert_equal(res[~df_infmask], res_ex_noinf)

    def test_logpdf_pdf(self):
        # 参考值是通过参考分布计算得到的，例如 mp.dps = 500; StudentT(df=df).logpdf(x), StudentT(df=df).pdf(x)
        x = [1, 1e3, 10, 1]
        df = [1e100, 1e50, 1e20, 1]
        # 参考值：对数概率密度和概率密度
        logpdf_ref = [-1.4189385332046727, -500000.9189385332,
                      -50.918938533204674, -1.8378770664093456]
        pdf_ref = [0.24197072451914334, 0,
                   7.69459862670642e-23, 0.15915494309189535]
        # 断言 t 分布的对数概率密度在给定 x 和 df 下与参考值 logpdf_ref 相近
        assert_allclose(stats.t.logpdf(x, df), logpdf_ref, rtol=1e-14)
        # 断言 t 分布的概率密度在给定 x 和 df 下与参考值 pdf_ref 相近
        assert_allclose(stats.t.pdf(x, df), pdf_ref, rtol=1e-14)
class TestRvDiscrete:
    # 在每个测试方法执行前设置随机种子，确保测试结果可重复
    def setup_method(self):
        np.random.seed(1234)

    # 测试随机变量生成函数的正确性
    def test_rvs(self):
        # 定义离散随机变量可能的状态和其对应的概率
        states = [-1, 0, 1, 2, 3, 4]
        probability = [0.0, 0.3, 0.4, 0.0, 0.3, 0.0]
        samples = 1000
        # 创建一个离散随机变量对象
        r = stats.rv_discrete(name='sample', values=(states, probability))
        # 生成指定数量的随机样本
        x = r.rvs(size=samples)
        # 断言生成的样本是 NumPy 数组
        assert isinstance(x, np.ndarray)

        # 验证生成的样本在每个状态上的频率与期望概率接近
        for s, p in zip(states, probability):
            assert abs(sum(x == s)/float(samples) - p) < 0.05

        # 再次生成单个随机样本，并验证其类型为整数
        x = r.rvs()
        assert np.issubdtype(type(x), np.integer)

    # 测试离散随机变量的熵计算
    def test_entropy(self):
        # 基本的熵计算测试
        pvals = np.array([0.25, 0.45, 0.3])
        p = stats.rv_discrete(values=([0, 1, 2], pvals))
        expected_h = -sum(xlogy(pvals, pvals))
        h = p.entropy()
        assert_allclose(h, expected_h)

        # 特殊情况下的熵计算测试
        p = stats.rv_discrete(values=([0, 1, 2], [1.0, 0, 0]))
        h = p.entropy()
        assert_equal(h, 0.0)

    # 测试离散随机变量的概率质量函数
    def test_pmf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        # 验证离散随机变量在多个点的概率质量函数值
        x = [[1., 4.],
             [3., 2]]
        assert_allclose(rv.pmf(x),
                        [[0.5, 0.2],
                         [0., 0.3]], atol=1e-14)

    # 测试离散随机变量的累积分布函数
    def test_cdf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        # 验证离散随机变量在多个点的累积分布函数值
        x_values = [-2, 1., 1.1, 1.5, 2.0, 3.0, 4, 5]
        expected = [0, 0.5, 0.5, 0.5, 0.8, 0.8, 1, 1]
        assert_allclose(rv.cdf(x_values), expected, atol=1e-14)

        # 同时检查单个点作为参数时的累积分布函数值
        assert_allclose([rv.cdf(xx) for xx in x_values],
                        expected, atol=1e-14)

    # 测试离散随机变量的分位数函数
    def test_ppf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        # 验证离散随机变量在多个概率点的分位数函数值
        q_values = [0.1, 0.5, 0.6, 0.8, 0.9, 1.]
        expected = [1, 1, 2, 2, 4, 4]
        assert_allclose(rv.ppf(q_values), expected, atol=1e-14)

        # 同时检查单个概率值作为参数时的分位数函数值
        assert_allclose([rv.ppf(q) for q in q_values],
                        expected, atol=1e-14)

    # 测试特殊情况下离散随机变量的累积分布函数和分位数函数的连续性
    def test_cdf_ppf_next(self):
        # 从 test_discrete_basic 复制和特殊情况处理
        vals = ([1, 2, 4, 7, 8], [0.1, 0.2, 0.3, 0.3, 0.1])
        rv = stats.rv_discrete(values=vals)

        # 验证离散随机变量的分位数函数和累积分布函数的连续性
        assert_array_equal(rv.ppf(rv.cdf(rv.xk[:-1]) + 1e-8),
                           rv.xk[1:])

    # 测试多维离散随机变量的期望计算
    def test_multidimension(self):
        xk = np.arange(12).reshape((3, 4))
        pk = np.array([[0.1, 0.1, 0.15, 0.05],
                       [0.1, 0.1, 0.05, 0.05],
                       [0.1, 0.1, 0.05, 0.05]])
        rv = stats.rv_discrete(values=(xk, pk))

        # 验证多维离散随机变量的期望计算结果
        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)
    # 定义测试函数，用于测试不良输入情况下的行为
    def test_bad_input(self):
        # 设置离散随机变量的可能取值和概率
        xk = [1, 2, 3]
        pk = [0.5, 0.5]
        # 断言当传入这样的不良输入时，会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 更改概率列表为非法值列表
        pk = [1, 2, 3]
        # 再次断言传入这样的不良输入会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 更改概率列表为包含非法概率值的列表
        pk = [0.5, 1.2, -0.7]
        # 再次断言传入这样的不良输入会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 更改概率列表为和大于1的列表
        xk = [1, 2, 3, 4, 5]
        pk = [0.3, 0.3, 0.3, 0.3, -0.2]
        # 再次断言传入这样的不良输入会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 设置重复的离散随机变量的可能取值
        xk = [1, 1]
        pk = [0.5, 0.5]
        # 再次断言传入这样的不良输入会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

    # 定义测试函数，用于测试不同形状的离散随机变量样本的行为
    def test_shape_rv_sample(self):
        # 为 gh-9565 添加测试

        # 不匹配的二维输入
        xk, pk = np.arange(4).reshape((2, 2)), np.full((2, 3), 1/6)
        # 断言当传入这样的不良输入时，会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 元素数量相同但形状不兼容
        xk, pk = np.arange(6).reshape((3, 2)), np.full((2, 3), 1/6)
        # 再次断言传入这样的不良输入会引发 ValueError 异常
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # 形状相同 => 无错误
        xk, pk = np.arange(6).reshape((3, 2)), np.full((3, 2), 1/6)
        # 使用 assert_equal 断言确保当传入合法输入时，不会引发异常，并且计算得到的概率质量函数值为期望值的近似
        assert_equal(stats.rv_discrete(values=(xk, pk)).pmf(0), 1/6)

    # 定义测试函数，用于测试期望值计算的准确性
    def test_expect1(self):
        # 设置离散随机变量的可能取值和概率
        xk = [1, 2, 4, 6, 7, 11]
        pk = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        # 使用给定的取值和概率创建离散随机变量对象
        rv = stats.rv_discrete(values=(xk, pk))

        # 断言离散随机变量对象的期望值接近于通过手动计算得到的值，设置容差为 1e-14
        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)
    def test_expect2(self):
        # 定义一个包含概率分布的离散随机变量 rv
        # y 是离散随机变量的取值列表
        y = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
             1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
             1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0,
             2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0,
             3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0,
             4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0]

        # py 是对应 y 的概率列表
        py = [0.0004, 0.0, 0.0033, 0.006500000000000001, 0.0, 0.0,
              0.004399999999999999, 0.6862, 0.0, 0.0, 0.0,
              0.00019999999999997797, 0.0006000000000000449,
              0.024499999999999966, 0.006400000000000072,
              0.0043999999999999595, 0.019499999999999962,
              0.03770000000000007, 0.01759999999999995, 0.015199999999999991,
              0.018100000000000005, 0.04500000000000004, 0.0025999999999999357,
              0.0, 0.0041000000000001036, 0.005999999999999894,
              0.0042000000000000925, 0.0050000000000000044,
              0.0041999999999999815, 0.0004999999999999449,
              0.009199999999999986, 0.008200000000000096,
              0.0, 0.0, 0.0046999999999999265, 0.0019000000000000128,
              0.0006000000000000449, 0.02510000000000001, 0.0,
              0.007199999999999984, 0.0, 0.012699999999999934, 0.0, 0.0,
              0.008199999999999985, 0.005600000000000049, 0.0]

        # 根据 y 和 py 创建离散随机变量 rv
        rv = stats.rv_discrete(values=(y, py))

        # 检查期望值是否接近均值，容差为 1e-14
        assert_allclose(rv.expect(), rv.mean(), atol=1e-14)

        # 检查期望值是否接近加权平均值，容差为 1e-14
        assert_allclose(rv.expect(),
                        sum(v * w for v, w in zip(y, py)), atol=1e-14)

        # 还检查二阶矩的期望值是否接近加权平方平均值，容差为 1e-14
        assert_allclose(rv.expect(lambda x: x**2),
                        sum(v**2 * w for v, w in zip(y, py)), atol=1e-14)
class TestSkewCauchy:
    def test_cauchy(self):
        # 创建一个包含100个均匀分布在[-5, 5]之间的数值的数组
        x = np.linspace(-5, 5, 100)
        # 断言 skewcauchy 分布的概率密度函数在 a=0 时与 cauchy 分布的概率密度函数几乎相等
        assert_array_almost_equal(stats.skewcauchy.pdf(x, a=0),
                                  stats.cauchy.pdf(x))
        # 断言 skewcauchy 分布的累积分布函数在 a=0 时与 cauchy 分布的累积分布函数几乎相等
        assert_array_almost_equal(stats.skewcauchy.cdf(x, a=0),
                                  stats.cauchy.cdf(x))
        # 断言 skewcauchy 分布的分位数函数在 a=0 时与 cauchy 分布的分位数函数几乎相等
        assert_array_almost_equal(stats.skewcauchy.ppf(x, a=0),
                                  stats.cauchy.ppf(x))

    def test_skewcauchy_R(self):
        # 下面是 R 语言代码，用于生成 lambda 和 x 的值
        # options(digits=16)
        # library(sgt)
        # # lmbda, x contain the values generated for a, x below
        # lmbda <- c(0.0976270078546495, 0.430378732744839, 0.2055267521432877,
        #            0.0897663659937937, -0.15269040132219, 0.2917882261333122,
        #            -0.12482557747462, 0.7835460015641595, 0.9273255210020589,
        #            -0.2331169623484446)
        # x <- c(2.917250380826646, 0.2889491975290444, 0.6804456109393229,
        #        4.25596638292661, -4.289639418021131, -4.1287070029845925,
        #        -4.797816025596743, 3.32619845547938, 2.7815675094985046,
        #        3.700121482468191)
        # 使用固定的随机种子以确保可重复性
        np.random.seed(0)
        # 生成一个包含10个在[-1, 1]之间的随机数数组
        a = np.random.rand(10) * 2 - 1
        # 生成一个包含10个在[-5, 5]之间的随机数数组
        x = np.random.rand(10) * 10 - 5
        # 预先计算的 PDF 值列表
        pdf = [0.039473975217333909, 0.305829714049903223, 0.24140158118994162,
               0.019585772402693054, 0.021436553695989482, 0.00909817103867518,
               0.01658423410016873, 0.071083288030394126, 0.103250045941454524,
               0.013110230778426242]
        # 预先计算的 CDF 值列表
        cdf = [0.87426677718213752, 0.37556468910780882, 0.59442096496538066,
               0.91304659850890202, 0.09631964100300605, 0.03829624330921733,
               0.08245240578402535, 0.72057062945510386, 0.62826415852515449,
               0.95011308463898292]
        # 断言 skewcauchy 分布的概率密度函数与预期的 pdf 值列表几乎相等
        assert_allclose(stats.skewcauchy.pdf(x, a), pdf)
        # 断言 skewcauchy 分布的累积分布函数与预期的 cdf 值列表几乎相等
        assert_allclose(stats.skewcauchy.cdf(x, a), cdf)
        # 断言 skewcauchy 分布的分位数函数在给定 cdf 值时与原始 x 值几乎相等
        assert_allclose(stats.skewcauchy.ppf(cdf, a), x)


class TestJFSkewT:
    def test_compare_t(self):
        # 验证当 a=b 时，jf_skew_t 分布可以回归到自由度为 2a 的 t 分布
        a = b = 5
        df = a * 2
        # 指定一组用于测试的 x 值
        x = [-1.0, 0.0, 1.0, 2.0]
        # 指定一组用于测试的 q 值
        q = [0.0, 0.1, 0.25, 0.75, 0.90, 1.0]

        # 创建 jf_skew_t 分布对象
        jf = stats.jf_skew_t(a, b)
        # 创建 t 分布对象
        t = stats.t(df)

        # 断言 jf_skew_t 分布的概率密度函数与 t 分布的概率密度函数几乎相等
        assert_allclose(jf.pdf(x), t.pdf(x))
        # 断言 jf_skew_t 分布的累积分布函数与 t 分布的累积分布函数几乎相等
        assert_allclose(jf.cdf(x), t.cdf(x))
        # 断言 jf_skew_t 分布的分位数函数与 t 分布的分位数函数几乎相等
        assert_allclose(jf.ppf(q), t.ppf(q))
        # 断言 jf_skew_t 分布的一些统计值与 t 分布的对应统计值几乎相等
        assert_allclose(jf.stats('mvsk'), t.stats('mvsk'))

    @pytest.fixture
    def gamlss_pdf_data(self):
        """Sample data points computed using the `ST5` distribution from the
        GAMLSS package in R. The pdf has been calculated for (a,b)=(2,3),
        (a,b)=(8,4), and (a,b)=(12,13) for x in `np.linspace(-10, 10, 41)`.

        N.B. the `ST5` distribution in R uses an alternative parameterization
        in terms of nu and tau, where:
            - nu = (a - b) / (a * b * (a + b)) ** 0.5
            - tau = 2 / (a + b)
        """
        # 加载预先计算的数据，包含了不同参数(a, b)下的 x, pdf, a, b 值
        data = np.load(
            Path(__file__).parent / "data/jf_skew_t_gamlss_pdf_data.npy"
        )
        # 返回一个记录数组，用于存储 x, pdf, a, b 四个字段的数据
        return np.rec.fromarrays(data, names="x,pdf,a,b")

    @pytest.mark.parametrize("a,b", [(2, 3), (8, 4), (12, 13)])
    def test_compare_with_gamlss_r(self, gamlss_pdf_data, a, b):
        """Compare the pdf with a table of reference values. The table of
        reference values was produced using R, where the Jones and Faddy skew
        t distribution is available in the GAMLSS package as `ST5`.
        """
        # 从预先计算的数据中选择特定参数 (a, b) 对应的数据点
        data = gamlss_pdf_data[
            (gamlss_pdf_data["a"] == a) & (gamlss_pdf_data["b"] == b)
        ]
        # 分别获取选择出的 x 和 pdf 值
        x, pdf = data["x"], data["pdf"]
        # 断言预测的 pdf 与通过 stats.jf_skew_t(a, b).pdf(x) 计算的 pdf 接近，相对误差小于 1e-12
        assert_allclose(pdf, stats.jf_skew_t(a, b).pdf(x), rtol=1e-12)
# Test data for TestSkewNorm.test_noncentral_moments()
# The expected noncentral moments were computed by Wolfram Alpha.
# In Wolfram Alpha, enter
#    SkewNormalDistribution[0, 1, a] moment
# with `a` replaced by the desired shape parameter.  In the results, there
# should be a table of the first four moments. Click on "More" to get more
# moments.  The expected moments start with the first moment (order = 1).
_skewnorm_noncentral_moments = [
    # Tuple for shape parameter 2
    (2, [2*np.sqrt(2/(5*np.pi)),        # First moment
         1,                             # Second moment
         22/5*np.sqrt(2/(5*np.pi)),     # Third moment
         3,                             # Fourth moment
         446/25*np.sqrt(2/(5*np.pi)),   # Fifth moment
         15,                            # Sixth moment
         2682/25*np.sqrt(2/(5*np.pi)),  # Seventh moment
         105,                           # Eighth moment
         107322/125*np.sqrt(2/(5*np.pi))]),  # Ninth moment
    # Tuple for shape parameter 0.1
    (0.1, [np.sqrt(2/(101*np.pi)),
           1,
           302/101*np.sqrt(2/(101*np.pi)),
           3,
           (152008*np.sqrt(2/(101*np.pi)))/10201,
           15,
           (107116848*np.sqrt(2/(101*np.pi)))/1030301,
           105,
           (97050413184*np.sqrt(2/(101*np.pi)))/104060401]),
    # Tuple for shape parameter -3
    (-3, [-3/np.sqrt(5*np.pi),
          1,
          -63/(10*np.sqrt(5*np.pi)),
          3,
          -2529/(100*np.sqrt(5*np.pi)),
          15,
          -30357/(200*np.sqrt(5*np.pi)),
          105,
          -2428623/(2000*np.sqrt(5*np.pi)),
          945,
          -242862867/(20000*np.sqrt(5*np.pi)),
          10395,
          -29143550277/(200000*np.sqrt(5*np.pi)),
          135135]),
]


class TestSkewNorm:
    # Setup method for initializing random state
    def setup_method(self):
        self.rng = check_random_state(1234)

    # Test case for verifying skewnorm.pdf behaves as norm.pdf when skewness is 0
    def test_normal(self):
        x = np.linspace(-5, 5, 100)
        assert_array_almost_equal(stats.skewnorm.pdf(x, a=0),
                                  stats.norm.pdf(x))

    # Test case for verifying shape and size of generated samples from skewnorm.rvs
    def test_rvs(self):
        shape = (3, 4, 5)
        x = stats.skewnorm.rvs(a=0.75, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)

        x = stats.skewnorm.rvs(a=-3, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)

    # Test case for verifying computed moments match expected moments within tolerance
    def test_moments(self):
        # Generating samples and calculating moments for positive skewness
        X = stats.skewnorm.rvs(a=4, size=int(1e6), loc=5, scale=2,
                               random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)

        # Generating samples and calculating moments for negative skewness
        X = stats.skewnorm.rvs(a=-4, size=int(1e6), loc=5, scale=2,
                               random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=-4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)
    def test_pdf_large_x(self):
        # 测试大的 x 值下的概率密度函数值
        # Triples are [x, a, logpdf(x, a)].  These values were computed
        # using Log[PDF[SkewNormalDistribution[0, 1, a], x]] in Wolfram Alpha.
        # 计算这些值时使用了 Wolfram Alpha 中 SkewNormalDistribution[0, 1, a] 的概率密度函数的对数
        logpdfvals = [
            [40, -1, -1604.834233366398515598970],
            [40, -1/2, -1004.142946723741991369168],
            [40, 0, -800.9189385332046727417803],
            [40, 1/2, -800.2257913526447274323631],
            [-40, -1/2, -800.2257913526447274323631],
            [-2, 1e7, -2.000000000000199559727173e14],
            [2, -1e7, -2.000000000000199559727173e14],
        ]
        for x, a, logpdfval in logpdfvals:
            # 计算 SkewNormal 分布的对数概率密度函数值
            logp = stats.skewnorm.logpdf(x, a)
            # 使用 assert_allclose 断言计算值与预期值的接近程度
            assert_allclose(logp, logpdfval, rtol=1e-8)

    def test_cdf_large_x(self):
        # 测试大的 x 值下的累积分布函数值
        # Regression test for gh-7746.
        # The x values are large enough that the closest 64 bit floating
        # point representation of the exact CDF is 1.0.
        # x 值足够大，使得最接近的 64 位浮点数表示的累积分布函数精确值为 1.0
        p = stats.skewnorm.cdf([10, 20, 30], -1)
        # 使用 assert_allclose 断言计算值与预期值的接近程度
        assert_allclose(p, np.ones(3), rtol=1e-14)
        p = stats.skewnorm.cdf(25, 2.5)
        # 使用 assert_allclose 断言计算值与预期值的接近程度
        assert_allclose(p, 1.0, rtol=1e-14)

    def test_cdf_sf_small_values(self):
        # 测试小的 x 值下的累积分布函数和生存函数值
        # Triples are [x, a, cdf(x, a)].  These values were computed
        # using CDF[SkewNormalDistribution[0, 1, a], x] in Wolfram Alpha.
        # 计算这些值时使用了 Wolfram Alpha 中 SkewNormalDistribution[0, 1, a] 的累积分布函数值
        cdfvals = [
            [-8, 1, 3.870035046664392611e-31],
            [-4, 2, 8.1298399188811398e-21],
            [-2, 5, 1.55326826787106273e-26],
            [-9, -1, 2.257176811907681295e-19],
            [-10, -4, 1.523970604832105213e-23],
        ]
        for x, a, cdfval in cdfvals:
            # 计算 SkewNormal 分布的累积分布函数值
            p = stats.skewnorm.cdf(x, a)
            # 使用 assert_allclose 断言计算值与预期值的接近程度
            assert_allclose(p, cdfval, rtol=1e-8)
            # 对于 SkewNormal 分布，sf(-x, -a) = cdf(x, a) 成立
            p = stats.skewnorm.sf(-x, -a)
            # 使用 assert_allclose 断言计算值与预期值的接近程度
            assert_allclose(p, cdfval, rtol=1e-8)

    @pytest.mark.parametrize('a, moments', _skewnorm_noncentral_moments)
    def test_noncentral_moments(self, a, moments):
        # 测试 SkewNormal 分布的非中心矩
        for order, expected in enumerate(moments, start=1):
            # 计算指定阶数的非中心矩
            mom = stats.skewnorm.moment(order, a)
            # 使用 assert_allclose 断言计算值与预期值的接近程度
            assert_allclose(mom, expected, rtol=1e-14)
    # 定义一个测试方法，用于测试 skewnorm 分布的拟合功能
    def test_fit(self):
        # 使用特定种子生成随机数生成器对象
        rng = np.random.default_rng(4609813989115202851)

        # 设置 skewnorm 分布的参数：a=-2, loc=3.5, scale=0.5
        a, loc, scale = -2, 3.5, 0.5
        # 创建 skewnorm 分布对象
        dist = stats.skewnorm(a, loc, scale)
        # 从分布中生成随机样本
        rvs = dist.rvs(size=100, random_state=rng)

        # 测试最大似然估计（MLE）仍然遵循给定的猜测和固定参数
        a2, loc2, scale2 = stats.skewnorm.fit(rvs, -1.5, floc=3)
        a3, loc3, scale3 = stats.skewnorm.fit(rvs, -1.6, floc=3)
        # 断言固定参数 loc 被尊重
        assert loc2 == loc3 == 3
        # 断言不同的猜测会导致略微不同的结果
        assert a2 != a3
        # 拟合质量在其他地方测试

        # 测试方法 of moments（MoM）拟合遵循固定参数，接受（但忽略）猜测
        a4, loc4, scale4 = stats.skewnorm.fit(rvs, 3, fscale=3, method='mm')
        # 断言 scale 被设置为 3
        assert scale4 == 3
        # 因为 scale 被固定，只有均值和偏度会被匹配
        dist4 = stats.skewnorm(a4, loc4, scale4)
        # 计算分布的均值和偏度
        res = dist4.stats(moments='ms')
        ref = np.mean(rvs), stats.skew(rvs)
        # 断言均值和偏度匹配
        assert_allclose(res, ref)

        # 测试当数据的偏度超过 skewnorm 的最大偏度时的行为
        rvs2 = stats.pareto.rvs(1, size=100, random_state=rng)

        # MLE 仍然有效
        res = stats.skewnorm.fit(rvs2)
        assert np.all(np.isfinite(res))

        # MoM 适合方差和偏度
        a5, loc5, scale5 = stats.skewnorm.fit(rvs2, method='mm')
        # 断言 shape 参数 a5 是无限的
        assert np.isinf(a5)
        # 分布结构不允许无限的形状参数进入 _stats；它会直接生成 NaN。手动计算矩。
        m, v = np.mean(rvs2), np.var(rvs2)
        # 断言均值和方差匹配
        assert_allclose(m, loc5 + scale5 * np.sqrt(2/np.pi))
        assert_allclose(v, scale5**2 * (1 - 2 / np.pi))

        # 测试当数据符号变化时，MLE 和 MoM 的行为如预期
        a6p, loc6p, scale6p = stats.skewnorm.fit(rvs, method='mle')
        a6m, loc6m, scale6m = stats.skewnorm.fit(-rvs, method='mle')
        # 断言相反数据的拟合参数关系
        assert_allclose([a6m, loc6m, scale6m], [-a6p, -loc6p, scale6p])
        a7p, loc7p, scale7p = stats.skewnorm.fit(rvs, method='mm')
        a7m, loc7m, scale7m = stats.skewnorm.fit(-rvs, method='mm')
        # 断言相反数据的拟合参数关系
        assert_allclose([a7m, loc7m, scale7m], [-a7p, -loc7p, scale7p])
    def test_fit_gh19332(self):
        # 当数据的偏度很高时，`skewnorm.fit`会退回到具有不良偏度参数猜测的通用 `fit` 行为。
        # 测试这是否得到改进；当样本高度偏斜时，`skewnorm.fit` 现在更擅长找到全局最优解。参见 gh-19332。
        x = np.array([-5, -1, 1 / 100_000] + 12 * [1] + [5])

        # 使用 `skewnorm.fit` 对数据 `x` 进行参数拟合
        params = stats.skewnorm.fit(x)
        # 计算使用拟合参数 `params` 的负对数似然函数值
        res = stats.skewnorm.nnlf(params, x)

        # 比较使用重写的拟合方法和通用的拟合方法
        # `res` 应该约为 32.01，而通用拟合的结果为 32.64 更差。
        # 如果通用拟合方法改进了，应删除此断言（参见 gh-19333）。
        params_super = stats.skewnorm.fit(x, superfit=True)
        ref = stats.skewnorm.nnlf(params_super, x)
        assert res < ref - 0.5

        # 比较重写的拟合方法与 `stats.fit` 的结果
        rng = np.random.default_rng(9842356982345693637)
        bounds = {'a': (-5, 5), 'loc': (-10, 10), 'scale': (1e-16, 10)}

        # 自定义优化器函数，使用差分进化算法进行拟合
        def optimizer(fun, bounds):
            return differential_evolution(fun, bounds, seed=rng)

        # 使用 `stats.fit` 对 `skewnorm` 分布和数据 `x` 进行拟合
        fit_result = stats.fit(stats.skewnorm, x, bounds, optimizer=optimizer)
        # 使用 `np.testing.assert_allclose` 检查参数拟合结果的近似程度
        np.testing.assert_allclose(params, fit_result.params, rtol=1e-4)

    def test_ppf(self):
        # gh-20124 报告 Boost 的 ppf 在高偏度时出现问题
        # 参考值是使用 Wolfram Alpha 计算的 N[InverseCDF[SkewNormalDistribution[0, 1, 500], 1/100], 14]。
        # 使用 `assert_allclose` 检查 `stats.skewnorm.ppf` 函数的返回值是否接近预期值
        assert_allclose(stats.skewnorm.ppf(0.01, 500), 0.012533469508013, rtol=1e-13)
class TestExpon:
    # 测试指数分布 PDF 在 0 处的取值是否为 1
    def test_zero(self):
        assert_equal(stats.expon.pdf(0), 1)

    # 回归测试，验证指数分布 CDF 在接近 0 处的取值
    def test_tail(self):  # Regression test for ticket 807
        assert_equal(stats.expon.cdf(1e-18), 1e-18)
        assert_equal(stats.expon.isf(stats.expon.sf(40)), 40)

    # 测试处理包含 NaN 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_nan_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.expon.fit, x)

    # 测试处理包含 Inf 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_inf_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.expon.fit, x)


class TestNorm:
    # 测试处理包含 NaN 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_nan_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.norm.fit, x)

    # 测试处理包含 Inf 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_inf_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.norm.fit, x)

    # 测试对于不正确的关键字参数是否引发 TypeError
    def test_bad_keyword_arg(self):
        x = [1, 2, 3]
        assert_raises(TypeError, stats.norm.fit, x, plate="shrimp")

    # 使用 pytest 的参数化功能测试 delta_cdf 方法的准确性
    @pytest.mark.parametrize('loc', [0, 1])
    def test_delta_cdf(self, loc):
        # 预期值是通过 mpmath 计算得出的
        expected = 1.910641809677555e-28
        delta = stats.norm._delta_cdf(11+loc, 12+loc, loc=loc)
        assert_allclose(delta, expected, rtol=1e-13)
        delta = stats.norm._delta_cdf(-(12+loc), -(11+loc), loc=-loc)
        assert_allclose(delta, expected, rtol=1e-13)


class TestUniform:
    """gh-10300"""
    # 测试处理包含 NaN 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_nan_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.uniform.fit, x)

    # 测试处理包含 Inf 的情况是否引发 ValueError，相关问题见 gh-issue 10300
    def test_inf_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.uniform.fit, x)


class TestExponNorm:
    def test_moments(self):
        # Some moment test cases based on non-loc/scaled formula
        
        # 定义一个函数，计算指定参数下的统计矩
        def get_moms(lam, sig, mu):
            # 查看维基百科上的公式，这里是指数修正高斯分布的公式
            opK2 = 1.0 + 1 / (lam*sig)**2
            exp_skew = 2 / (lam * sig)**3 * opK2**(-1.5)
            exp_kurt = 6.0 * (1 + (lam * sig)**2)**(-2)
            return [mu + 1/lam, sig*sig + 1.0/(lam*lam), exp_skew, exp_kurt]

        # 设置测试用例的参数
        mu, sig, lam = 0, 1, 1
        K = 1.0 / (lam * sig)
        # 使用指定参数调用 exponnorm 的 stats 函数计算统计值，并断言与自定义函数 get_moms 的返回值几乎相等
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        
        mu, sig, lam = -3, 2, 0.1
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        
        mu, sig, lam = 0, 3, 1
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        
        mu, sig, lam = -5, 11, 3.5
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))

    def test_nan_raises_error(self):
        # see gh-issue 10300
        # 创建包含 NaN 值的 NumPy 数组，并断言调用 exponnorm 的 fit 函数会引发 ValueError 异常
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_inf_raises_error(self):
        # see gh-issue 10300
        # 创建包含 Infinity 值的 NumPy 数组，并断言调用 exponnorm 的 fit 函数会引发 ValueError 异常
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_extremes_x(self):
        # Test for extreme values against overflows
        # 测试在极端值下的 PDF 计算，期望得到接近零的结果
        assert_almost_equal(stats.exponnorm.pdf(-900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(-900, 0.01), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 0.01), 0.0)

    # Expected values for the PDF were computed with mpmath, with
    # the following function, and with mpmath.mp.dps = 50.
    #
    #   def exponnorm_stdpdf(x, K):
    #       x = mpmath.mpf(x)
    #       K = mpmath.mpf(K)
    #       t1 = mpmath.exp(1/(2*K**2) - x/K)
    #       erfcarg = -(x - 1/K)/mpmath.sqrt(2)
    #       t2 = mpmath.erfc(erfcarg)
    #       return t1 * t2 / (2*K)
    #
    @pytest.mark.parametrize('x, K, expected',
                             [(20, 0.01, 6.90010764753618e-88),
                              (1, 0.01, 0.24438994313247364),
                              (-1, 0.01, 0.23955149623472075),
                              (-20, 0.01, 4.6004708690125477e-88),
                              (10, 1, 7.48518298877006e-05),
                              (10, 10000, 9.990005048283775e-05)])
    def test_std_pdf(self, x, K, expected):
        # 使用 pytest 的参数化功能测试 exponnorm 的 PDF 计算结果是否与预期值接近
        assert_allclose(stats.exponnorm.pdf(x, K), expected, rtol=5e-12)
    # 用于测试指数正态分布的累积分布函数（CDF）的期望值，使用了mpmath库和高精度计算
    @pytest.mark.parametrize('x, K, scale, expected',
                             [[0, 0.01, 1, 0.4960109760186432],
                              [-5, 0.005, 1, 2.7939945412195734e-07],
                              [-1e4, 0.01, 100, 0.0],
                              [-1e4, 0.01, 1000, 6.920401854427357e-24],
                              [5, 0.001, 1, 0.9999997118542392]])
    def test_cdf_small_K(self, x, K, scale, expected):
        # 计算给定参数下的累积分布函数值
        p = stats.exponnorm.cdf(x, K, scale=scale)
        # 如果期望值为0.0，则断言p应为0.0
        if expected == 0.0:
            assert p == 0.0
        else:
            # 否则，使用相对误差进行断言
            assert_allclose(p, expected, rtol=1e-13)

    # 用于测试指数正态分布的生存函数（SF）的期望值，同样使用了mpmath库和高精度计算
    @pytest.mark.parametrize('x, K, scale, expected',
                             [[10, 0.01, 1, 8.474702916146657e-24],
                              [2, 0.005, 1, 0.02302280664231312],
                              [5, 0.005, 0.5, 8.024820681931086e-24],
                              [10, 0.005, 0.5, 3.0603340062892486e-89],
                              [20, 0.005, 0.5, 0.0],
                              [-3, 0.001, 1, 0.9986545205566117]])
    def test_sf_small_K(self, x, K, scale, expected):
        # 计算给定参数下的生存函数值
        p = stats.exponnorm.sf(x, K, scale=scale)
        # 如果期望值为0.0，则断言p应为0.0
        if expected == 0.0:
            assert p == 0.0
        else:
            # 否则，使用相对误差进行断言
            assert_allclose(p, expected, rtol=5e-13)
class TestGenExpon:
    def test_pdf_unity_area(self):
        from scipy.integrate import simpson
        # 导入 simpson 函数用于数值积分
        # 生成广义指数分布的概率密度函数并计算其在 [0, 10) 范围内的积分，应接近 1
        p = stats.genexpon.pdf(np.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert_almost_equal(simpson(p, dx=0.01), 1, 1)

    def test_cdf_bounds(self):
        # 检查广义指数分布的累积分布函数在 [0, 10) 范围内始终为非负数且不超过 1
        cdf = stats.genexpon.cdf(np.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert np.all((0 <= cdf) & (cdf <= 1))

    # 下列数据中的概率值由 mpmath 计算得出。
    # 例如，使用以下脚本计算：
    #     from mpmath import mp
    #     mp.dps = 80
    #     x = mp.mpf('15.0')
    #     a = mp.mpf('1.0')
    #     b = mp.mpf('2.0')
    #     c = mp.mpf('1.5')
    #     print(float(mp.exp((-a-b)*x + (b/c)*-mp.expm1(-c*x))))
    # 输出为
    #     1.0859444834514553e-19
    @pytest.mark.parametrize('x, p, a, b, c',
                             [(15, 1.0859444834514553e-19, 1, 2, 1.5),
                              (0.25, 0.7609068232534623, 0.5, 2, 3),
                              (0.25, 0.09026661397565876, 9.5, 2, 0.5),
                              (0.01, 0.9753038265071597, 2.5, 0.25, 0.5),
                              (3.25, 0.0001962824553094492, 2.5, 0.25, 0.5),
                              (0.125, 0.9508674287164001, 0.25, 5, 0.5)])
    def test_sf_isf(self, x, p, a, b, c):
        # 检查广义指数分布的生存函数 sf 的计算结果与预期概率值 p 的接近程度
        sf = stats.genexpon.sf(x, a, b, c)
        assert_allclose(sf, p, rtol=2e-14)
        # 检查广义指数分布的逆生存函数 isf 的计算结果与预期值 x 的接近程度
        isf = stats.genexpon.isf(p, a, b, c)
        assert_allclose(isf, x, rtol=2e-14)

    # 下列数据中的概率值由 mpmath 计算得出。
    @pytest.mark.parametrize('x, p, a, b, c',
                             [(0.25, 0.2390931767465377, 0.5, 2, 3),
                              (0.25, 0.9097333860243412, 9.5, 2, 0.5),
                              (0.01, 0.0246961734928403, 2.5, 0.25, 0.5),
                              (3.25, 0.9998037175446906, 2.5, 0.25, 0.5),
                              (0.125, 0.04913257128359998, 0.25, 5, 0.5)])
    def test_cdf_ppf(self, x, p, a, b, c):
        # 检查广义指数分布的累积分布函数 cdf 的计算结果与预期概率值 p 的接近程度
        cdf = stats.genexpon.cdf(x, a, b, c)
        assert_allclose(cdf, p, rtol=2e-14)
        # 检查广义指数分布的分位数函数 ppf 的计算结果与预期值 x 的接近程度
        ppf = stats.genexpon.ppf(p, a, b, c)
        assert_allclose(ppf, x, rtol=2e-14)


class TestTruncexpon:

    def test_sf_isf(self):
        # 参考值通过参考分布计算，例如 mp.dps = 50; TruncExpon(b=b).sf(x)
        b = [20, 100]
        x = [19.999999, 99.999999]
        ref = [2.0611546593828472e-15, 3.7200778266671455e-50]
        # 检查截尾指数分布的生存函数 sf 的计算结果与预期参考值 ref 的接近程度
        assert_allclose(stats.truncexpon.sf(x, b), ref, rtol=1.5e-10)
        # 检查截尾指数分布的逆生存函数 isf 的计算结果与预期值 x 的接近程度
        assert_allclose(stats.truncexpon.isf(ref, b), x, rtol=1e-12)


class TestExponpow:
    def test_tail(self):
        # 检查指数幂分布的累积分布函数 cdf 在接近零处的计算结果
        assert_almost_equal(stats.exponpow.cdf(1e-10, 2.), 1e-20)
        # 检查指数幂分布的逆生存函数 isf 在给定生存函数 sf 的结果上的正确性
        assert_almost_equal(stats.exponpow.isf(stats.exponpow.sf(5, .8), .8),
                            5)
    def test_pmf(self):
        # 比较结果与 R 语言的概率质量函数值
        k = np.arange(-10, 15)  # 创建一个从 -10 到 14 的整数数组
        mu1, mu2 = 10, 5  # 设置参数 mu1 和 mu2 的值
        skpmfR = np.array(
                   [4.2254582961926893e-005, 1.1404838449648488e-004,
                    2.8979625801752660e-004, 6.9177078182101231e-004,
                    1.5480716105844708e-003, 3.2412274963433889e-003,
                    6.3373707175123292e-003, 1.1552351566696643e-002,
                    1.9606152375042644e-002, 3.0947164083410337e-002,
                    4.5401737566767360e-002, 6.1894328166820688e-002,
                    7.8424609500170578e-002, 9.2418812533573133e-002,
                    1.0139793148019728e-001, 1.0371927988298846e-001,
                    9.9076583077406091e-002, 8.8546660073089561e-002,
                    7.4187842052486810e-002, 5.8392772862200251e-002,
                    4.3268692953013159e-002, 3.0248159818374226e-002,
                    1.9991434305603021e-002, 1.2516877303301180e-002,
                    7.4389876226229707e-003])

        # 使用断言比较 stats.skellam.pmf(k, mu1, mu2) 的结果与 skpmfR 数组的值，精度为 15 位小数
        assert_almost_equal(stats.skellam.pmf(k, mu1, mu2), skpmfR, decimal=15)

    def test_cdf(self):
        # 比较结果与 R 语言的累积分布函数值，精度为 5 位小数
        k = np.arange(-10, 15)  # 创建一个从 -10 到 14 的整数数组
        mu1, mu2 = 10, 5  # 设置参数 mu1 和 mu2 的值
        skcdfR = np.array(
                   [6.4061475386192104e-005, 1.7810985988267694e-004,
                    4.6790611790020336e-004, 1.1596768997212152e-003,
                    2.7077485103056847e-003, 5.9489760066490718e-003,
                    1.2286346724161398e-002, 2.3838698290858034e-002,
                    4.3444850665900668e-002, 7.4392014749310995e-002,
                    1.1979375231607835e-001, 1.8168808048289900e-001,
                    2.6011268998306952e-001, 3.5253150251664261e-001,
                    4.5392943399683988e-001, 5.5764871387982828e-001,
                    6.5672529695723436e-001, 7.4527195703032389e-001,
                    8.1945979908281064e-001, 8.7785257194501087e-001,
                    9.2112126489802404e-001, 9.5136942471639818e-001,
                    9.7136085902200120e-001, 9.8387773632530240e-001,
                    9.9131672394792536e-001])

        # 使用断言比较 stats.skellam.cdf(k, mu1, mu2) 的结果与 skcdfR 数组的值，精度为 5 位小数
        assert_almost_equal(stats.skellam.cdf(k, mu1, mu2), skcdfR, decimal=5)

    def test_extreme_mu2(self):
        # 检查是否解决了 gh-17916 报告的大 mu2 值导致的崩溃问题
        x, mu1, mu2 = 0, 1, 4820232647677555.0  # 设置参数 x, mu1 和 mu2 的值
        # 使用 assert_allclose 断言比较 stats.skellam.pmf(x, mu1, mu2) 的结果与 0 的接近程度，误差限为 1e-16
        assert_allclose(stats.skellam.pmf(x, mu1, mu2), 0, atol=1e-16)
        # 使用 assert_allclose 断言比较 stats.skellam.cdf(x, mu1, mu2) 的结果与 1 的接近程度，误差限为 1e-16
        assert_allclose(stats.skellam.cdf(x, mu1, mu2), 1, atol=1e-16)
class TestLognorm:
    def test_pdf(self):
        # Regression test for Ticket #1471: avoid nan with 0/0 situation
        # Also make sure there are no warnings at x=0, cf gh-5202
        # 使用警告捕获来检测运行时警告，并确保在 x=0 时没有警告，参见 gh-5202
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # 计算对数正态分布的概率密度函数，验证避免 0/0 情况下的结果
            pdf = stats.lognorm.pdf([0, 0.5, 1], 1)
            assert_array_almost_equal(pdf, [0.0, 0.62749608, 0.39894228])

    def test_logcdf(self):
        # Regression test for gh-5940: sf et al would underflow too early
        # 针对 gh-5940 的回归测试：确保 sf 等函数不会在太早的时候下溢
        x2, mu, sigma = 201.68, 195, 0.149
        # 使用 lognorm 分布的生存函数（sf），验证其与正态分布对数的生存函数的一致性
        assert_allclose(stats.lognorm.sf(x2-mu, s=sigma),
                        stats.norm.sf(np.log(x2-mu)/sigma))
        # 使用 lognorm 分布的对数生存函数（logsf），验证其与正态分布对数的对数生存函数的一致性
        assert_allclose(stats.lognorm.logsf(x2-mu, s=sigma),
                        stats.norm.logsf(np.log(x2-mu)/sigma))

    @pytest.fixture(scope='function')
    def rng(self):
        # 返回一个随机数生成器对象，用于测试函数的随机数生成
        return np.random.default_rng(1234)

    @pytest.mark.parametrize("rvs_shape", [.1, 2])
    @pytest.mark.parametrize("rvs_loc", [-2, 0, 2])
    @pytest.mark.parametrize("rvs_scale", [.2, 1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [e for e in product((False, True), repeat=3)
                              if False in e])
    @np.errstate(invalid="ignore")
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        # 生成服从对数正态分布的随机数据，用于最大似然估计的比较优化器的测试
        data = stats.lognorm.rvs(size=100, s=rvs_shape, scale=rvs_scale,
                                 loc=rvs_loc, random_state=rng)

        kwds = {}
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        # 如果分析路线中的某些代码路径使用了数值优化，则数值结果可能等于分析结果
        _assert_less_or_close_loglike(stats.lognorm, data, **kwds,
                                      maybe_identical=True)

    def test_isf(self):
        # reference values were computed via the reference distribution, e.g.
        # mp.dps = 100;
        # LogNormal(s=s).isf(q=0.1, guess=0)
        # LogNormal(s=s).isf(q=2e-10, guess=100)
        # 计算对数正态分布的逆生存函数，验证其与参考分布的一致性
        s = 0.954
        q = [0.1, 2e-10, 5e-20, 6e-40]
        ref = [3.3960065375794937, 390.07632793595974, 5830.5020828128445,
               287872.84087457904]
        assert_allclose(stats.lognorm.isf(q, s), ref, rtol=1e-14)


class TestBeta:
    def test_logpdf(self):
        # Regression test for Ticket #1326: avoid nan with 0*log(0) situation
        # 针对 Ticket #1326 的回归测试：避免 0*log(0) 情况下的结果为 nan
        # 计算 beta 分布的对数概率密度函数，验证其正确性
        logpdf = stats.beta.logpdf(0, 1, 0.5)
        assert_almost_equal(logpdf, -0.69314718056)
        logpdf = stats.beta.logpdf(0, 0.5, 1)
        assert_almost_equal(logpdf, np.inf)
    # 定义一个单元测试方法，测试 beta 分布的 logpdf 方法的正确性
    def test_logpdf_ticket_1866(self):
        # 设置 beta 分布的参数 alpha 和 beta
        alpha, beta = 267, 1472
        # 创建一个包含浮点数的 NumPy 数组 x
        x = np.array([0.2, 0.5, 0.6])
        # 根据给定的 alpha 和 beta 创建 beta 分布对象 b
        b = stats.beta(alpha, beta)
        # 断言 beta 分布的 logpdf 方法计算出的结果总和等于预期值 -1201.699061824062
        assert_allclose(b.logpdf(x).sum(), -1201.699061824062)
        # 断言 beta 分布的 pdf 方法计算出的结果与 exp(logpdf) 的结果相等
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))

    # 定义一个单元测试方法，测试 beta 分布的 fit 方法在使用不正确的关键字参数时是否会抛出 TypeError
    def test_fit_bad_keyword_args(self):
        # 创建一个包含浮点数的列表 x
        x = [0.1, 0.5, 0.6]
        # 断言调用 beta 分布的 fit 方法时传递 plate="shrimp" 会抛出 TypeError
        assert_raises(TypeError, stats.beta.fit, x, floc=0, fscale=1,
                      plate="shrimp")

    # 定义一个单元测试方法，测试 beta 分布的 fit 方法在重复指定固定参数时是否会抛出 ValueError
    def test_fit_duplicated_fixed_parameter(self):
        # 创建一个包含浮点数的列表 x
        x = [0.1, 0.5, 0.6]
        # 断言调用 beta 分布的 fit 方法时同时指定 fa=0.5 和 fix_a=0.5 会抛出 ValueError
        assert_raises(ValueError, stats.beta.fit, x, fa=0.5, fix_a=0.5)

    # 定义一个单元测试方法，测试 Boost 实现的 beta 分布解决 GitHub 问题 gh-12635 的准确性
    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12635(self):
        # 确认 Boost 实现的 beta 分布解决 GitHub 问题 gh-12635
        # 根据 R 代码确认：
        # options(digits=16)
        # p = 0.9999999999997369
        # a = 75.0
        # b = 66334470.0
        # print(qbeta(p, a, b))
        p, a, b = 0.9999999999997369, 75.0, 66334470.0
        # 断言 stats.beta.ppf 方法计算出的结果与预期值 2.343620802982393e-06 接近
        assert_allclose(stats.beta.ppf(p, a, b), 2.343620802982393e-06)

    # 定义一个单元测试方法，测试 Boost 实现的 beta 分布解决 GitHub 问题 gh-12794 的准确性
    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12794(self):
        # 确认 Boost 实现的 beta 分布解决 GitHub 问题 gh-12794
        # 根据 R 代码确认：
        # options(digits=16)
        # p = 1e-11
        # count_list = c(10,100,1000)
        # print(qbeta(1-p, count_list + 1, 100000 - count_list))
        inv_R = np.array([0.0004944464889611935,
                          0.0018360586912635726,
                          0.0122663919942518351])
        count_list = np.array([10, 100, 1000])
        p = 1e-11
        # 使用 stats.beta.isf 方法计算 beta 分布的分位数 inv
        inv = stats.beta.isf(p, count_list + 1, 100000 - count_list)
        # 断言计算出的 inv 数组与预期的 inv_R 数组非常接近
        assert_allclose(inv, inv_R)
        # 使用 stats.beta.sf 方法计算 beta 分布的生存函数 res
        res = stats.beta.sf(inv, count_list + 1, 100000 - count_list)
        # 断言计算出的 res 数组与预期的概率 p 非常接近
        assert_allclose(res, p)

    # 定义一个单元测试方法，测试 Boost 实现的 beta 分布解决 GitHub 问题 gh-12796 的准确性
    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12796(self):
        # 确认 Boost 实现的 beta 分布解决 GitHub 问题 gh-12796
        alpha_2 = 5e-6
        count_ = np.arange(1, 20)
        nobs = 100000
        q, a, b = 1 - alpha_2, count_ + 1, nobs - count_
        # 使用 stats.beta.ppf 方法计算 beta 分布的分位数 inv
        inv = stats.beta.ppf(q, a, b)
        # 使用 stats.beta.cdf 方法计算 beta 分布的累积分布函数 res
        res = stats.beta.cdf(inv, a, b)
        # 断言计算出的 res 数组与 1 - alpha_2 非常接近
        assert_allclose(res, 1 - alpha_2)
    def test_endpoints(self):
        # 确认在 b<1 时，boost 的 beta 分布在 x=1 处返回无穷大
        a, b = 1, 0.5
        assert_equal(stats.beta.pdf(1, a, b), np.inf)

        # 确认在 a<1 时，boost 的 beta 分布在 x=0 处返回无穷大
        a, b = 0.2, 3
        assert_equal(stats.beta.pdf(0, a, b), np.inf)

        # 确认当 a=1, b=5 时，boost 的 beta 分布在 x=0 处返回 5
        a, b = 1, 5
        assert_equal(stats.beta.pdf(0, a, b), 5)
        assert_equal(stats.beta.pdf(1e-310, a, b), 5)

        # 确认当 a=5, b=1 时，boost 的 beta 分布在 x=1 处返回 5
        a, b = 5, 1
        assert_equal(stats.beta.pdf(1, a, b), 5)
        assert_equal(stats.beta.pdf(1-1e-310, a, b), 5)

    @pytest.mark.xfail(IS_PYPY, reason="Does not convert boost warning")
    def test_boost_eval_issue_14606(self):
        # 测试 boost 的问题 14606：当 q=0.995, a=1.0e11, b=1.0e13 时
        q, a, b = 0.995, 1.0e11, 1.0e13
        with pytest.warns(RuntimeWarning):
            stats.beta.ppf(q, a, b)

    @pytest.mark.parametrize('method', [stats.beta.ppf, stats.beta.isf])
    @pytest.mark.parametrize('a, b', [(1e-310, 12.5), (12.5, 1e-310)])
    def test_beta_ppf_with_subnormal_a_b(self, method, a, b):
        # 回归测试 gh-17444：当 a 或 b 是亚正常数时，beta.ppf(p, a, b) 和 beta.isf(p, a, b)
        # 可能导致分段错误。在此处我们接受可能的 OverflowError 或返回值，
        # 目的是验证调用不会触发分段错误。
        p = 0.9
        try:
            method(p, a, b)
        except OverflowError:
            # 当 Boost 1.80 或更早版本中 Boost 的双精度提升策略为假时，会引发 OverflowError
            # 参考：
            #   https://github.com/boostorg/math/issues/882
            #   https://github.com/boostorg/math/pull/883
            # 一旦我们使用了修复后的 Boost 版本，可以移除这个 try-except 包装器并直接调用函数。
            pass

    # entropy accuracy was confirmed using the following mpmath function
    # entropy 的准确性使用以下 mpmath 函数进行了确认
    # from mpmath import mp
    # mp.dps = 50
    # def beta_entropy_mpmath(a, b):
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     entropy = mp.log(mp.beta(a, b)) - (a - 1) * mp.digamma(a) -\
    #              (b - 1) * mp.digamma(b) + (a + b -2) * mp.digamma(a + b)
    #     return float(entropy)

    @pytest.mark.parametrize('a, b, ref',
                             [(0.5, 0.5, -0.24156447527049044),
                              (0.001, 1, -992.0922447210179),
                              (1, 10000, -8.210440371976183),
                              (100000, 100000, -5.377247470132859)])
    # 定义测试函数，用于验证 beta 分布的熵计算是否正确
    def test_entropy(self, a, b, ref):
        # 使用 assert_allclose 断言函数，验证 stats.beta(a, b).entropy() 的返回值是否与 ref 接近
        assert_allclose(stats.beta(a, b).entropy(), ref)

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同参数组合下的熵计算结果
    @pytest.mark.parametrize(
        "a, b, ref, tol",
        [
            (1, 10, -1.4025850929940458, 1e-14),
            (10, 20, -1.0567887388936708, 1e-13),
            (4e6, 4e6+20, -7.221686009678741, 1e-9),
            (5e6, 5e6+10, -7.333257022834638, 1e-8),
            (1e10, 1e10+20, -11.133707703130474, 1e-11),
            (1e50, 1e50+20, -57.185409562486385, 1e-15),
            (2, 1e10, -21.448635265288925, 1e-11),
            (2, 1e20, -44.47448619497938, 1e-14),
            (2, 1e50, -113.55203898480075, 1e-14),
            (5, 1e10, -20.87226777401971, 1e-10),
            (5, 1e20, -43.89811870326017, 1e-14),
            (5, 1e50, -112.97567149308153, 1e-14),
            (10, 1e10, -20.489796752909477, 1e-9),
            (10, 1e20, -43.51564768139993, 1e-14),
            (10, 1e50, -112.59320047122131, 1e-14),
            (1e20, 2, -44.47448619497938, 1e-14),
            (1e20, 5, -43.89811870326017, 1e-14),
            (1e50, 10, -112.59320047122131, 1e-14),
        ]
    )
    # 定义极端情况下的熵计算测试函数
    def test_extreme_entropy(self, a, b, ref, tol):
        # 使用 assert_allclose 断言函数，验证 stats.beta(a, b).entropy() 的返回值是否与 ref 接近，设置允许的相对误差为 tol
        assert_allclose(stats.beta(a, b).entropy(), ref, rtol=tol)
class TestBetaPrime:
    # the test values are used in test_cdf_gh_17631 / test_ppf_gh_17631
    # They are computed with mpmath. Example:
    # from mpmath import mp
    # mp.dps = 50
    # a, b = mp.mpf(0.05), mp.mpf(0.1)
    # x = mp.mpf(1e22)
    # float(mp.betainc(a, b, 0.0, x/(1+x), regularized=True))
    # note: we use the values computed by the cdf to test whether
    # ppf(cdf(x)) == x (up to a small tolerance)
    # since the ppf can be very sensitive to small variations of the input,
    # it can be required to generate the test case for the ppf separately,
    # see self.test_ppf
    cdf_vals = [
        (1e22, 100.0, 0.05, 0.8973027435427167),      # Test values for cumulative distribution function (cdf)
        (1e10, 100.0, 0.05, 0.5911548582766262),
        (1e8, 0.05, 0.1, 0.9467768090820048),
        (1e8, 100.0, 0.05, 0.4852944858726726),
        (1e-10, 0.05, 0.1, 0.21238845427095),
        (1e-10, 1.5, 1.5, 1.697652726007973e-15),
        (1e-10, 0.05, 100.0, 0.40884514172337383),
        (1e-22, 0.05, 0.1, 0.053349567649287326),
        (1e-22, 1.5, 1.5, 1.6976527263135503e-33),
        (1e-22, 0.05, 100.0, 0.10269725645728331),
        (1e-100, 0.05, 0.1, 6.7163126421919795e-06),
        (1e-100, 1.5, 1.5, 1.6976527263135503e-150),
        (1e-100, 0.05, 100.0, 1.2928818587561651e-05),
    ]

    def test_logpdf(self):
        alpha, beta = 267, 1472
        x = np.array([0.2, 0.5, 0.6])
        b = stats.betaprime(alpha, beta)
        assert_(np.isfinite(b.logpdf(x)).all())     # Check that all log pdf values are finite
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))  # Compare pdf values with exp(logpdf) for numerical consistency

    def test_cdf(self):
        # regression test for gh-4030: Implementation of
        # scipy.stats.betaprime.cdf()
        x = stats.betaprime.cdf(0, 0.2, 0.3)     # Compute cdf value for specific parameters
        assert_equal(x, 0.0)    # Assert that cdf(0, 0.2, 0.3) equals 0.0

        alpha, beta = 267, 1472
        x = np.array([0.2, 0.5, 0.6])
        cdfs = stats.betaprime.cdf(x, alpha, beta)   # Compute cdfs for array x with parameters alpha, beta
        assert_(np.isfinite(cdfs).all())    # Check that all cdf values are finite

        # check the new cdf implementation vs generic one:
        gen_cdf = stats.rv_continuous._cdf_single
        cdfs_g = [gen_cdf(stats.betaprime, val, alpha, beta) for val in x]
        assert_allclose(cdfs, cdfs_g, atol=0, rtol=2e-12)   # Assert that the specific cdfs are close to generic ones

    # The expected values for test_ppf() were computed with mpmath, e.g.
    #
    #   from mpmath import mp
    #   mp.dps = 125
    #   p = 0.01
    #   a, b = 1.25, 2.5
    #   x = mp.findroot(lambda t: mp.betainc(a, b, x1=0, x2=t/(1+t),
    #                                        regularized=True) - p,
    #                   x0=(0.01, 0.011), method='secant')
    #   print(float(x))
    #
    # prints
    #
    #   0.01080162700956614
    #
    @pytest.mark.parametrize(
        'p, a, b, expected',
        [(0.010, 1.25, 2.5, 0.01080162700956614),    # Test parameters and expected values for ppf
         (1e-12, 1.25, 2.5, 1.0610141996279122e-10),
         (1e-18, 1.25, 2.5, 1.6815941817974941e-15),
         (1e-17, 0.25, 7.0, 1.0179194531881782e-69),
         (0.375, 0.25, 7.0, 0.002036820346115211),
         (0.9978811466052919, 0.05, 0.1, 1.0000000000001218e22),]
    )
    # 测试统计库中的 betaprime 分布的 percent point function (ppf)，即累积分布函数的逆运算
    def test_ppf(self, p, a, b, expected):
        x = stats.betaprime.ppf(p, a, b)
        # 断言计算得到的逆运算结果与预期结果非常接近
        assert_allclose(x, expected, rtol=1e-14)

    # 使用参数化测试标记，测试 betaprime 分布的 ppf，验证逆运算结果是否正确
    @pytest.mark.parametrize('x, a, b, p', cdf_vals)
    def test_ppf_gh_17631(self, x, a, b, p):
        # 断言计算得到的逆运算结果与预期值非常接近
        assert_allclose(stats.betaprime.ppf(p, a, b), x, rtol=2e-14)

    # 使用参数化测试标记，测试 betaprime 分布的累积分布函数 (cdf)，验证计算结果是否正确
    @pytest.mark.parametrize(
        'x, a, b, expected',
        cdf_vals + [
            (1e10, 1.5, 1.5, 0.9999999999999983),
            (1e10, 0.05, 0.1, 0.9664184367890859),
            (1e22, 0.05, 0.1, 0.9978811466052919),
        ])
    def test_cdf_gh_17631(self, x, a, b, expected):
        # 断言计算得到的累积分布函数结果与预期值非常接近
        assert_allclose(stats.betaprime.cdf(x, a, b), expected, rtol=1e-14)

    # 使用参数化测试标记，测试 betaprime 分布在极端尾部情况下的累积分布函数 (cdf)
    @pytest.mark.parametrize(
        'x, a, b, expected',
        [(1e50, 0.05, 0.1, 0.9999966641709545),
         (1e50, 100.0, 0.05, 0.995925162631006)])
    def test_cdf_extreme_tails(self, x, a, b, expected):
        # 对于极端情况，验证累积分布函数的结果仍然小于1，并且与预期值非常接近
        y = stats.betaprime.cdf(x, a, b)
        assert y < 1.0
        assert_allclose(y, expected, rtol=2e-5)

    # 测试 betaprime 分布的 survival function (sf)，即 1 - cdf(x)
    def test_sf(self):
        # 参考值是通过使用参考分布计算得到的
        # 使用不同参数进行计算
        a = [5, 4, 2, 0.05, 0.05, 0.05, 0.05, 100.0, 100.0, 0.05, 0.05,
             0.05, 1.5, 1.5]
        b = [3, 2, 1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 100.0, 100.0,
             100.0, 1.5, 1.5]
        x = [1e10, 1e20, 1e30, 1e22, 1e-10, 1e-22, 1e-100, 1e22, 1e10,
             1e-10, 1e-22, 1e-100, 1e10, 1e-10]
        ref = [3.4999999979e-29, 9.999999999994357e-40, 1.9999999999999998e-30,
               0.0021188533947081017, 0.78761154572905, 0.9466504323507127,
               0.9999932836873578, 0.10269725645728331, 0.40884514172337383,
               0.5911548582766262, 0.8973027435427167, 0.9999870711814124,
               1.6976527260079727e-15, 0.9999999999999983]
        # 计算 betaprime 分布的 survival function 值，并断言其与参考值非常接近
        sf_values = stats.betaprime.sf(x, a, b)
        assert_allclose(sf_values, ref, rtol=1e-12)

    # 测试 betaprime 分布的拟合函数，验证是否解决了 gh-18274 报告的问题
    def test_fit_stats_gh18274(self):
        # gh-18274 报告拟合 `betaprime` 到数据时会发出误报警告，检查现在是否不再发出这些警告
        # 对一些数据进行拟合，并检查是否有警告输出
        stats.betaprime.fit([0.1, 0.25, 0.3, 1.2, 1.6], floc=0, fscale=1)
        # 计算 `betaprime` 分布的一些统计值，并无返回结果，只验证函数能否正常运行
        stats.betaprime(a=1, b=1).stats('mvsk')
    # 定义测试函数 test_moment_gh18634，用于测试 gh-18634 中的问题
    def test_moment_gh18634(self):
        # 在测试 gh-18634 时发现 `betaprime` 在高阶时刻（moment）抛出了 NotImplementedError。
        # 确保此问题已解决。参数是任意选择的，但位于时刻顺序（5）的两侧，以测试 `_lazywhere` 的两个分支。
        # 参考值由 Mathematica 等软件生成，例如 `Moment[BetaPrimeDistribution[2,7],5]`
        ref = [np.inf, 0.867096912929055]
        # 调用 stats 模块中的 betaprime 分布，计算其时刻（moment）为 5 的结果
        res = stats.betaprime(2, [4.2, 7.1]).moment(5)
        # 使用 assert_allclose 函数验证计算结果与参考值的接近程度
        assert_allclose(res, ref)
class TestGamma:
    # Gamma 分布的测试类

    def test_pdf(self):
        # 比较结果与 R 语言的几个测试用例
        pdf = stats.gamma.pdf(90, 394, scale=1./5)
        assert_almost_equal(pdf, 0.002312341)

        pdf = stats.gamma.pdf(3, 10, scale=1./5)
        assert_almost_equal(pdf, 0.1620358)

    def test_logpdf(self):
        # Ticket #1326 的回归测试：避免在 0*log(0) 情况下出现 NaN
        logpdf = stats.gamma.logpdf(0, 1)
        assert_almost_equal(logpdf, 0)

    def test_fit_bad_keyword_args(self):
        # 测试 stats.gamma.fit 函数对于错误的关键字参数的处理
        x = [0.1, 0.5, 0.6]
        assert_raises(TypeError, stats.gamma.fit, x, floc=0, plate="shrimp")

    def test_isf(self):
        # 当概率非常小时的测试用例。参见 gh-13664。
        # 可以使用 mpmath 进行检查预期值。
        #
        # mpmath 中，生存函数 sf(x, k) 的计算方法如下：
        #
        #     mpmath.gammainc(k, x, mpmath.inf, regularized=True)
        #
        # 这里有：
        #
        # >>> mpmath.mp.dps = 60
        # >>> float(mpmath.gammainc(1, 39.14394658089878, mpmath.inf,
        # ...                       regularized=True))
        # 9.99999999999999e-18
        # >>> float(mpmath.gammainc(100, 330.6557590436547, mpmath.inf,
        #                           regularized=True))
        # 1.000000000000028e-50
        #
        assert np.isclose(stats.gamma.isf(1e-17, 1),
                          39.14394658089878, atol=1e-14)
        assert np.isclose(stats.gamma.isf(1e-50, 100),
                          330.6557590436547, atol=1e-13)

    @pytest.mark.parametrize('scale', [1.0, 5.0])
    def test_delta_cdf(self, scale):
        # 使用 mpmath 计算的预期值：
        #
        # >>> import mpmath
        # >>> mpmath.mp.dps = 150
        # >>> cdf1 = mpmath.gammainc(3, 0, 245, regularized=True)
        # >>> cdf2 = mpmath.gammainc(3, 0, 250, regularized=True)
        # >>> float(cdf2 - cdf1)
        # 1.1902609356171962e-102
        #
        delta = stats.gamma._delta_cdf(scale*245, scale*250, 3, scale=scale)
        assert_allclose(delta, 1.1902609356171962e-102, rtol=1e-13)

    @pytest.mark.parametrize('a, ref, rtol',
                             [(1e-4, -9990.366610819761, 1e-15),
                              (2, 1.5772156649015328, 1e-15),
                              (100, 3.7181819485047463, 1e-13),
                              (1e4, 6.024075385026086, 1e-15),
                              (1e18, 22.142204370151084, 1e-15),
                              (1e100, 116.54819318290696, 1e-15)])
    def test_entropy(self, a, ref, rtol):
        # 使用 mpmath 计算的预期熵值：
        # from mpmath import mp
        # mp.dps = 500
        # def gamma_entropy_reference(x):
        #     x = mp.mpf(x)
        #     return float(mp.digamma(x) * (mp.one - x) + x + mp.loggamma(x))
        #
        assert_allclose(stats.gamma.entropy(a), ref, rtol=rtol)

    @pytest.mark.parametrize("a", [1e-2, 1, 1e2])
    # 使用 pytest 的参数化装饰器，为测试用例指定不同的参数 loc
    @pytest.mark.parametrize("loc", [1e-2, 0, 1e2])
    # 使用 pytest 的参数化装饰器，为测试用例指定不同的参数 scale
    @pytest.mark.parametrize('scale', [1e-2, 1, 1e2])
    # 使用 pytest 的参数化装饰器，为测试用例指定不同的参数 fix_a
    @pytest.mark.parametrize('fix_a', [True, False])
    # 使用 pytest 的参数化装饰器，为测试用例指定不同的参数 fix_loc
    @pytest.mark.parametrize('fix_loc', [True, False])
    # 使用 pytest 的参数化装饰器，为测试用例指定不同的参数 fix_scale
    @pytest.mark.parametrize('fix_scale', [True, False])
    # 定义测试方法 test_fit_mm，参数为 a, loc, scale, fix_a, fix_loc, fix_scale
    def test_fit_mm(self, a, loc, scale, fix_a, fix_loc, fix_scale):
        # 使用指定种子创建随机数生成器对象 rng
        rng = np.random.default_rng(6762668991392531563)
        # 从 Gamma 分布生成数据，参数为 a, loc, scale，样本数为 100
        data = stats.gamma.rvs(a, loc=loc, scale=scale, size=100,
                               random_state=rng)

        # 初始化空字典 kwds
        kwds = {}
        # 如果 fix_a 为 True，则将 a 放入字典 kwds 中对应的键 'fa'
        if fix_a:
            kwds['fa'] = a
        # 如果 fix_loc 为 True，则将 loc 放入字典 kwds 中对应的键 'floc'
        if fix_loc:
            kwds['floc'] = loc
        # 如果 fix_scale 为 True，则将 scale 放入字典 kwds 中对应的键 'fscale'
        if fix_scale:
            kwds['fscale'] = scale
        # 计算自由参数个数 nfree
        nfree = 3 - len(kwds)

        # 如果 nfree 等于 0，表示所有参数都被固定，抛出 ValueError 异常
        if nfree == 0:
            error_msg = "All parameters fixed. There is nothing to optimize."
            # 使用 pytest 的上下文管理器，检查是否抛出预期的 ValueError 异常，并匹配特定错误消息
            with pytest.raises(ValueError, match=error_msg):
                stats.gamma.fit(data, method='mm', **kwds)
            return

        # 使用方法 'mm' 拟合数据，传入字典 kwds 作为参数
        theta = stats.gamma.fit(data, method='mm', **kwds)
        # 基于拟合参数创建 Gamma 分布对象 dist
        dist = stats.gamma(*theta)
        # 如果 nfree 大于等于 1，断言分布的均值接近数据的均值
        if nfree >= 1:
            assert_allclose(dist.mean(), np.mean(data))
        # 如果 nfree 大于等于 2，断言分布的二阶矩接近数据的二阶矩
        if nfree >= 2:
            assert_allclose(dist.moment(2), np.mean(data**2))
        # 如果 nfree 大于等于 3，断言分布的三阶矩接近数据的三阶矩
        if nfree >= 3:
            assert_allclose(dist.moment(3), np.mean(data**3))
def test_pdf_overflow_gh19616():
    # 确认解决了 PDF 中的中间溢出/下溢问题 gh19616
    # 从 R GeneralizedHyperbolic 库中引用的参考值
    # library(GeneralizedHyperbolic)
    # options(digits=16)
    # jitter = 1e-3
    # dnig(1, a=2**0.5 / jitter**2, b=1 / jitter**2)
    # 设置 jitter 值用于计算
    jitter = 1e-3
    # 使用 stats 模块中的 norminvgauss 函数计算 Z 值
    Z = stats.norminvgauss(2**0.5 / jitter**2, 1 / jitter**2, loc=0, scale=1)
    # 断言 Z 对应的概率密度函数值接近于给定值
    assert_allclose(Z.pdf(1.0), 282.0948446666433)


class TestDgamma:
    def test_pdf(self):
        # 使用 np.random.default_rng 创建随机数生成器对象 rng
        rng = np.random.default_rng(3791303244302340058)
        # 设置 size 为要检查的点数
        size = 10
        # 使用 rng 生成服从正态分布的随机数 x
        x = rng.normal(scale=10, size=size)
        # 使用 rng 生成在 [0, 10) 范围内均匀分布的随机数 a
        a = rng.uniform(high=10, size=size)
        # 计算 dgamma 分布的概率密度函数在 x 处的值
        res = stats.dgamma.pdf(x, a)
        # 计算 gamma 分布绝对值下的概率密度函数在 x 处的值
        ref = stats.gamma.pdf(np.abs(x), a) / 2
        # 断言 res 和 ref 数组的所有元素接近
        assert_allclose(res, ref)

        # 创建 dgamma 分布对象 dist
        dist = stats.dgamma(a)
        # 在 Linux - 32 位系统上，有时 assert_equal 会出现间歇性失败
        # 使用 assert_allclose 断言 dist 对象的概率密度函数在 x 处的值接近于 res 数组的对应元素
        assert_allclose(dist.pdf(x), res, rtol=5e-16)

    # 使用 mpmath 计算预期值。
    # 对于 x < 0，cdf(x, a) 是 mp.gammainc(a, -x, mp.inf, regularized=True)/2
    # 对于 x > 0，cdf(x, a) 是 (1 + mp.gammainc(a, 0, x, regularized=True))/2
    # 例如：
    #    from mpmath import mp
    #    mp.dps = 50
    #    print(float(mp.gammainc(1, 20, mp.inf, regularized=True)/2))
    # 输出
    #    1.030576811219279e-09
    @pytest.mark.parametrize('x, a, expected',
                             [(-20, 1, 1.030576811219279e-09),
                              (-40, 1, 2.1241771276457944e-18),
                              (-50, 5, 2.7248509914602648e-17),
                              (-25, 0.125, 5.333071920958156e-14),
                              (5, 1, 0.9966310265004573)])
    def test_cdf_ppf_sf_isf_tail(self, x, a, expected):
        # 计算 dgamma 分布的累积分布函数值 cdf
        cdf = stats.dgamma.cdf(x, a)
        # 断言 cdf 值接近于预期值 expected
        assert_allclose(cdf, expected, rtol=5e-15)
        # 计算 dgamma 分布的百分位点函数值 ppf
        ppf = stats.dgamma.ppf(expected, a)
        # 断言 ppf 值接近于 x
        assert_allclose(ppf, x, rtol=5e-15)
        # 计算 dgamma 分布的生存函数值 sf
        sf = stats.dgamma.sf(-x, a)
        # 断言 sf 值接近于预期值 expected
        assert_allclose(sf, expected, rtol=5e-15)
        # 计算 dgamma 分布的逆生存函数值 isf
        isf = stats.dgamma.isf(expected, a)
        # 断言 isf 值接近于 -x
        assert_allclose(isf, -x, rtol=5e-15)

    @pytest.mark.parametrize("a, ref",
                             [(1.5, 2.0541199559354117),
                              (1.3, 1.9357296377121247),
                              (1.1, 1.7856502333412134)])
    def test_entropy(self, a, ref):
        # 参考值使用 mpmath 计算：
        # def entropy_dgamma(a):
        #    def pdf(x):
        #        A = mp.one / (mp.mpf(2.) * mp.gamma(a))
        #        B = mp.fabs(x) ** (a - mp.one)
        #        C = mp.exp(-mp.fabs(x))
        #        h = A * B * C
        #        return h
        #
        #    return -mp.quad(lambda t: pdf(t) * mp.log(pdf(t)),
        #                    [-mp.inf, mp.inf])
        # 使用 assert_allclose 断言 stats.dgamma.entropy(a) 的计算值接近于参考值 ref
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-14)
    @pytest.mark.parametrize("a, ref",
                             [(1e-100, -1e+100),
                             (1e-10, -9999999975.858217),
                             (1e-5, -99987.37111657023),
                             (1e4, 6.717222565586032),
                             (1000000000000000.0, 19.38147391121996),
                             (1e+100, 117.2413403634669)])
    # 使用 pytest 的 parametrize 装饰器，定义了多组参数和对应的参考值，用于测试
    def test_entropy_entreme_values(self, a, ref):
        # 参考值是使用 mpmath 计算得到的：
        # from mpmath import mp
        # mp.dps = 500
        # def second_dgamma(a):
        #     a = mp.mpf(a)
        #     x_1 = a + mp.log(2) + mp.loggamma(a)
        #     x_2 = (mp.one - a) * mp.digamma(a)
        #     h = x_1 + x_2
        #     return h
        # 使用 assert_allclose 函数断言 stats.dgamma.entropy(a) 的返回值接近于 ref
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-10)

    # 测试 stats.dgamma.entropy 函数对数组输入的处理
    def test_entropy_array_input(self):
        # 创建一个 NumPy 数组
        x = np.array([1, 5, 1e20, 1e-5])
        # 计算 stats.dgamma.entropy 对整个数组 x 的结果
        y = stats.dgamma.entropy(x)
        # 逐个检查每个元素的计算结果是否符合预期
        for i in range(len(y)):
            assert y[i] == stats.dgamma.entropy(x[i])
class TestChi2:
    # 回归测试，精度改进后的验证，票号：1041，未验证
    def test_precision(self):
        # 断言：验证卡方分布概率密度函数在给定参数下的计算精度
        assert_almost_equal(stats.chi2.pdf(1000, 1000), 8.919133934753128e-003,
                            decimal=14)
        assert_almost_equal(stats.chi2.pdf(100, 100), 0.028162503162596778,
                            decimal=14)

    def test_ppf(self):
        # 预期值通过 mpmath 计算得出
        df = 4.8
        x = stats.chi2.ppf(2e-47, df)
        # 断言：验证卡方分布的累积分布函数的反函数在给定参数下的计算精度
        assert_allclose(x, 1.098472479575179840604902808e-19, rtol=1e-10)
        x = stats.chi2.ppf(0.5, df)
        assert_allclose(x, 4.15231407598589358660093156, rtol=1e-10)

        df = 13
        x = stats.chi2.ppf(2e-77, df)
        assert_allclose(x, 1.0106330688195199050507943e-11, rtol=1e-10)
        x = stats.chi2.ppf(0.1, df)
        assert_allclose(x, 7.041504580095461859307179763, rtol=1e-10)

    # 熵的参考值是用以下 mpmath 代码计算的
    # from mpmath import mp
    # mp.dps = 50
    # def chisq_entropy_mpmath(df):
    #     df = mp.mpf(df)
    #     half_df = 0.5 * df
    #     entropy = (half_df + mp.log(2) + mp.log(mp.gamma(half_df)) +
    #                (mp.one - half_df) * mp.digamma(half_df))
    #     return float(entropy)

    @pytest.mark.parametrize('df, ref',
                             [(1e-4, -19988.980448690163),
                              (1, 0.7837571104739337),
                              (100, 4.061397128938114),
                              (251, 4.525577254045129),
                              (1e15, 19.034900320939986)])
    def test_entropy(self, df, ref):
        # 断言：验证卡方分布的熵在给定自由度下的计算精度
        assert_allclose(stats.chi2(df).entropy(), ref, rtol=1e-13)


class TestGumbelL:
    # gh-6228
    def test_cdf_ppf(self):
        # 创建一个从-100到-4的均匀分布的数值序列
        x = np.linspace(-100, -4)
        # 计算并断言：验证 Gumbel 左侧分布的累积分布函数和其反函数的逆过程
        y = stats.gumbel_l.cdf(x)
        xx = stats.gumbel_l.ppf(y)
        assert_allclose(x, xx)

    def test_logcdf_logsf(self):
        # 创建一个从-100到-4的均匀分布的数值序列
        x = np.linspace(-100, -4)
        # 计算并断言：验证 Gumbel 左侧分布的对数累积分布函数和对数生存函数之间的关系
        y = stats.gumbel_l.logcdf(x)
        z = stats.gumbel_l.logsf(x)
        u = np.exp(y)
        v = -special.expm1(z)
        assert_allclose(u, v)

    def test_sf_isf(self):
        # 创建一个从-20到5的均匀分布的数值序列
        x = np.linspace(-20, 5)
        # 计算并断言：验证 Gumbel 左侧分布的生存函数和其反函数的逆过程
        y = stats.gumbel_l.sf(x)
        xx = stats.gumbel_l.isf(y)
        assert_allclose(x, xx)

    @pytest.mark.parametrize('loc', [-1, 1])
    def test_fit_fixed_param(self, loc):
        # 确保从 `gumbel_r.fit` 函数中正确反映固定位置的值
        # 参见 gh-12737 末尾的注释
        # 生成一个带有固定位置参数的 Gumbel 左侧分布的随机样本数据
        data = stats.gumbel_l.rvs(size=100, loc=loc)
        # 拟合数据，并断言：验证固定位置参数是否正确反映在 `gumbel_r.fit` 中
        fitted_loc, _ = stats.gumbel_l.fit(data, floc=loc)
        assert_equal(fitted_loc, loc)


class TestGumbelR:
    # 定义测试函数，测试 stats 模块中 Gumbel 分布的生存函数（右尾概率）
    def test_sf(self):
        # 使用 mpmath 计算预期值：
        #   >>> import mpmath
        #   >>> mpmath.mp.dps = 40
        #   >>> float(mpmath.mp.one - mpmath.exp(-mpmath.exp(-50)))
        #   1.9287498479639178e-22
        # 断言 stats.gumbel_r.sf(50) 的计算结果接近预期值 1.9287498479639178e-22，相对误差容差为 1e-14
        assert_allclose(stats.gumbel_r.sf(50), 1.9287498479639178e-22,
                        rtol=1e-14)

    # 定义测试函数，测试 stats 模块中 Gumbel 分布的反生存函数（逆右尾概率）
    def test_isf(self):
        # 使用 mpmath 计算预期值：
        #   >>> import mpmath
        #   >>> mpmath.mp.dps = 40
        #   >>> float(-mpmath.log(-mpmath.log(mpmath.mp.one - 1e-17)))
        #   39.14394658089878
        # 断言 stats.gumbel_r.isf(1e-17) 的计算结果接近预期值 39.14394658089878，相对误差容差为 1e-14
        assert_allclose(stats.gumbel_r.isf(1e-17), 39.14394658089878,
                        rtol=1e-14)
# 定义一个测试类 TestLevyStable，用于测试 levy_stable 生成器
class TestLevyStable:

    # 设置 levy_stable 生成器的默认参数的 pytest fixture，自动使用
    @pytest.fixture(autouse=True)
    def reset_levy_stable_params(self):
        """Setup default parameters for levy_stable generator"""
        # 设置参数化方式为 "S1"
        stats.levy_stable.parameterization = "S1"
        # 设置默认的累积分布函数计算方法为 "piecewise"
        stats.levy_stable.cdf_default_method = "piecewise"
        # 设置默认的概率密度函数计算方法为 "piecewise"
        stats.levy_stable.pdf_default_method = "piecewise"
        # 设置数值积分的精度阈值
        stats.levy_stable.quad_eps = stats._levy_stable._QUAD_EPS

    # 定义一个 pytest fixture nolan_pdf_sample_data，用于返回用于 pdf 计算的样本数据
    @pytest.fixture
    def nolan_pdf_sample_data(self):
        """Sample data points for pdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,
        and the equivalent for the right tail

        Typically inputs for stablec:

            stablec.exe <<
            1 # pdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        # 从文件加载 Nolan's stablec 生成的样本数据
        data = np.load(
            Path(__file__).parent / 'data/levy_stable/stable-Z1-pdf-sample-data.npy'
        )
        # 将数据转换为 numpy 的记录数组，字段分别为 'x', 'p', 'alpha', 'beta', 'pct'
        data = np.rec.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data
    def nolan_cdf_sample_data(self):
        """Sample data points for cdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,

        and the equivalent for the right tail

        Ideally, Nolan's output for CDF values should match the percentile
        from where they have been sampled from. Even more so as we extract
        percentile x positions from stablec too. However, we note at places
        Nolan's stablec will produce absolute errors in order of 1e-5. We
        compare against his calculations here. In future, once we less
        reliant on Nolan's paper we might switch to comparing directly at
        percentiles (those x values being produced from some alternative
        means).

        Typically inputs for stablec:

            stablec.exe <<
            2 # cdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        # Load sample data from a precomputed file related to Levy stable distributions
        data = np.load(
            Path(__file__).parent /
            'data/levy_stable/stable-Z1-cdf-sample-data.npy'
        )
        # Convert loaded data into a structured NumPy array for easier access
        data = np.rec.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data

    @pytest.fixture
    def nolan_loc_scale_sample_data(self):
        """Sample data where loc, scale are different from 0, 1

        Data extracted in similar way to pdf/cdf above using
        Nolan's stablec but set to an arbitrary location scale of
        (2, 3) for various important parameters alpha, beta and for
        parameterisations S0 and S1.
        """
        # Load sample data from a precomputed file for Levy stable distributions
        data = np.load(
            Path(__file__).parent /
            'data/levy_stable/stable-loc-scale-sample-data.npy'
        )
        return data

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "sample_size", [
            pytest.param(50), pytest.param(1500, marks=pytest.mark.slow)
        ]
    )
    @pytest.mark.parametrize("parameterization", ["S0", "S1"])
    @pytest.mark.parametrize(
        "alpha,beta", [(1.0, 0), (1.0, -0.5), (1.5, 0), (1.9, 0.5)]
    )
    @pytest.mark.parametrize("gamma,delta", [(1, 0), (3, 2)])
    def test_rvs(
            self,
            parameterization,
            alpha,
            beta,
            gamma,
            delta,
            sample_size,
    ):
        # Set the parameterization type for Levy stable distributions
        stats.levy_stable.parameterization = parameterization
        # Initialize a Levy stable distribution with specified parameters
        ls = stats.levy_stable(
            alpha=alpha, beta=beta, scale=gamma, loc=delta
        )
        # Perform a Kolmogorov-Smirnov test comparing sample data against the CDF
        _, p = stats.kstest(
            ls.rvs(size=sample_size, random_state=1234), ls.cdf
        )
        # Assert that the p-value from the K-S test is greater than 0.05
        assert p > 0.05

    @pytest.mark.xslow
    # 使用 pytest.mark.parametrize 装饰器为 test_rvs_alpha1 方法提供参数化测试，测试参数为 beta 取值为 0.5 和 1
    @pytest.mark.parametrize('beta', [0.5, 1])
    def test_rvs_alpha1(self, beta):
        """Additional test cases for rvs for alpha equal to 1."""
        # 设定随机种子
        np.random.seed(987654321)
        # 设定稳定分布的参数
        alpha = 1.0
        loc = 0.5
        scale = 1.5
        # 生成服从稳定分布的随机变量
        x = stats.levy_stable.rvs(alpha, beta, loc=loc, scale=scale,
                                  size=5000)
        # 对生成的随机变量进行 Kolmogorov-Smirnov 测试
        stat, p = stats.kstest(x, 'levy_stable',
                               args=(alpha, beta, loc, scale))
        # 断言检验 p 值是否大于 0.01
        assert p > 0.01

    # test_fit 方法，测试稳定分布的参数估计函数 _fitstart 的准确性
    def test_fit(self):
        # 构造数据，确保其百分位数与 McCulloch 1986 年的例子匹配
        x = [
            -.05413, -.05413, 0., 0., 0., 0., .00533, .00533, .00533, .00533,
            .00533, .03354, .03354, .03354, .03354, .03354, .05309, .05309,
            .05309, .05309, .05309
        ]
        # 使用稳定分布的参数估计函数 _fitstart 对数据进行估计
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        # 断言检验估计的参数 alpha1 是否接近于 1.48
        assert_allclose(alpha1, 1.48, rtol=0, atol=0.01)
        # 断言检验估计的参数 beta1 是否接近于 -0.22，精度为小数点后 2 位
        assert_almost_equal(beta1, -.22, 2)
        # 断言检验估计的参数 scale1 是否接近于 0.01717，精度为小数点后 4 位
        assert_almost_equal(scale1, 0.01717, 4)
        # 断言检验估计的参数 loc1 是否接近于 0.00233，精度为小数点后 2 位，由于 McCulloch86 中的舍入误差
        assert_almost_equal(
            loc1, 0.00233, 2
        )

        # 对 alpha=2 的情况进行覆盖测试
        x2 = x + [.05309, .05309, .05309, .05309, .05309]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        # 断言检验估计的参数 alpha2 是否等于 2
        assert_equal(alpha2, 2)
        # 断言检验估计的参数 beta2 是否等于 -1
        assert_equal(beta2, -1)
        # 断言检验估计的参数 scale2 是否接近于 0.02503，精度为小数点后 4 位
        assert_almost_equal(scale2, .02503, 4)
        # 断言检验估计的参数 loc2 是否接近于 0.03354，精度为小数点后 4 位
        assert_almost_equal(loc2, .03354, 4)

    # test_fit_rvs 方法，测试稳定分布参数估计与随机变量生成的一致性
    @pytest.mark.xfail(reason="Unknown problem with fitstart.")
    @pytest.mark.parametrize(
        "alpha,beta,delta,gamma",
        [
            (1.5, 0.4, 2, 3),
            (1.0, 0.4, 2, 3),
        ]
    )
    @pytest.mark.parametrize(
        "parametrization", ["S0", "S1"]
    )
    def test_fit_rvs(self, alpha, beta, delta, gamma, parametrization):
        """Test that fit agrees with rvs for each parametrization."""
        # 设置当前稳定分布的参数化方式
        stats.levy_stable.parametrization = parametrization
        # 生成符合指定参数的稳定分布的随机变量数据
        data = stats.levy_stable.rvs(
            alpha, beta, loc=delta, scale=gamma, size=10000, random_state=1234
        )
        # 使用稳定分布的参数估计函数 _fitstart 对生成的随机变量数据进行参数估计
        fit = stats.levy_stable._fitstart(data)
        # 检验估计的参数与设定的参数是否接近
        alpha_obs, beta_obs, delta_obs, gamma_obs = fit
        assert_allclose(
            [alpha, beta, delta, gamma],
            [alpha_obs, beta_obs, delta_obs, gamma_obs],
            rtol=0.01,
        )

    # test_fit_beta_flip 方法，测试 beta 的正负号对 loc 参数的影响
    def test_fit_beta_flip(self):
        # 确认 beta 的正负号影响 loc 参数，而不影响 alpha 或 scale
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        # 分别计算未取反和取反情况下的稳定分布参数估计
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x)
        # 断言检验未取反时 beta1 是否等于 1
        assert_equal(beta1, 1)
        # 断言检验 loc1 是否不等于 0
        assert loc1 != 0
        # 断言检验取反后 alpha2 是否接近于未取反时的 alpha1
        assert_almost_equal(alpha2, alpha1)
        # 断言检验取反后 beta2 是否接近于未取反时的 -beta1
        assert_almost_equal(beta2, -beta1)
        # 断言检验取反后 loc2 是否接近于未取反时的 -loc1
        assert_almost_equal(loc2, -loc1)
        # 断言检验取反后 scale2 是否接近于未取反时的 scale1
        assert_almost_equal(scale2, scale1)
    def test_fit_delta_shift(self):
        # 确认当数据发生偏移时，loc（位置参数）是否上下滑动。
        SHIFT = 1
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        # 获取适合 Levy stable 分布拟合的起始参数
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(-x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x + SHIFT)
        # 断言两次拟合的 alpha 参数近似相等
        assert_almost_equal(alpha2, alpha1)
        # 断言两次拟合的 beta 参数近似相等
        assert_almost_equal(beta2, beta1)
        # 断言 loc2 比 loc1 增加了 SHIFT
        assert_almost_equal(loc2, loc1 + SHIFT)
        # 断言两次拟合的 scale 参数近似相等
        assert_almost_equal(scale2, scale1)

    def test_fit_loc_extrap(self):
        # 确认当 alpha 接近 1 时，loc（位置参数）是否超出样本范围。
        x = [1, 1, 3, 3, 10, 10, 10, 30, 30, 140, 140]
        # 获取适合 Levy stable 分布拟合的起始参数
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        # 断言 alpha1 小于 1
        assert alpha1 < 1, f"Expected alpha < 1, got {alpha1}"
        # 断言 loc1 小于 x 中的最小值
        assert loc1 < min(x), f"Expected loc < {min(x)}, got {loc1}"

        x2 = [1, 1, 3, 3, 10, 10, 10, 30, 30, 130, 130]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        # 断言 alpha2 大于 1
        assert alpha2 > 1, f"Expected alpha > 1, got {alpha2}"
        # 断言 loc2 大于 x2 中的最大值
        assert loc2 > max(x2), f"Expected loc > {max(x2)}, got {loc2}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "pct_range,alpha_range,beta_range", [
            pytest.param(
                [.01, .5, .99],
                [.1, 1, 2],
                [-1, 0, .8],
            ),
            pytest.param(
                [.01, .05, .5, .95, .99],
                [.1, .5, 1, 1.5, 2],
                [-.9, -.5, 0, .3, .6, 1],
                marks=pytest.mark.slow
            ),
            pytest.param(
                [.01, .05, .1, .25, .35, .5, .65, .75, .9, .95, .99],
                np.linspace(0.1, 2, 20),
                np.linspace(-1, 1, 21),
                marks=pytest.mark.xslow,
            ),
        ]
    )
    def test_pdf_nolan_samples(
            self, nolan_pdf_sample_data, pct_range, alpha_range, beta_range
    ):
        # 测试 Levy stable 分布的概率密度函数对不同样本数据的样本集合
        pass

    @pytest.mark.parametrize(
        "pct_range,alpha_range,beta_range", [
            pytest.param(
                [.01, .5, .99],
                [.1, 1, 2],
                [-1, 0, .8],
            ),
            pytest.param(
                [.01, .05, .5, .95, .99],
                [.1, .5, 1, 1.5, 2],
                [-.9, -.5, 0, .3, .6, 1],
                marks=pytest.mark.slow
            ),
            pytest.param(
                [.01, .05, .1, .25, .35, .5, .65, .75, .9, .95, .99],
                np.linspace(0.1, 2, 20),
                np.linspace(-1, 1, 21),
                marks=pytest.mark.xslow,
            ),
        ]
    )
    def test_cdf_nolan_samples(
            self, nolan_cdf_sample_data, pct_range, alpha_range, beta_range
    ):
        # 测试 Levy stable 分布的累积分布函数对不同样本数据的样本集合
        pass

    @pytest.mark.parametrize("param", [0, 1])
    @pytest.mark.parametrize("case", ["pdf", "cdf"])
    def test_location_scale(
            self, nolan_loc_scale_sample_data, param, case
    ):
        # 测试 Levy stable 分布的位置和尺度参数对不同情况的样本集合
        pass
    ):
        """
        Tests for pdf and cdf where loc, scale are different from 0, 1
        """

        uname = platform.uname()
        is_linux_32 = uname.system == 'Linux' and "32bit" in platform.architecture()[0]
        # Test seems to be unstable (see gh-17839 for a bug report on Debian
        # i386), so skip it.
        # 如果运行在 Linux 32 位系统，并且是测试概率密度函数（pdf），则跳过该测试
        if is_linux_32 and case == 'pdf':
            pytest.skip("Test unstable on some platforms; see gh-17839, 17859")

        data = nolan_loc_scale_sample_data
        # 我们只对使用分段函数作为位置/缩放变换的方法进行测试
        stats.levy_stable.cdf_default_method = "piecewise"
        stats.levy_stable.pdf_default_method = "piecewise"

        subdata = data[data["param"] == param]
        stats.levy_stable.parameterization = f"S{param}"

        assert case in ["pdf", "cdf"]
        function = (
            stats.levy_stable.pdf if case == "pdf" else stats.levy_stable.cdf
        )

        v1 = function(
            subdata['x'], subdata['alpha'], subdata['beta'], scale=2, loc=3
        )
        # 断言计算得到的值与预期值在一定精度下相等
        assert_allclose(v1, subdata[case], 1e-5)

    @pytest.mark.parametrize(
        "method,decimal_places",
        [
            ['dni', 4],
            ['piecewise', 4],
        ]
    )
    def test_pdf_alpha_equals_one_beta_non_zero(self, method, decimal_places):
        """
        Sample points extracted from Tables and Graphs of Stable
        Probability Density Functions - Donald R Holt - 1973 - p 187.
        """
        xs = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )
        density = np.array(
            [
                .3183, .3096, .2925, .2622, .1591, .1587, .1599, .1635, .0637,
                .0729, .0812, .0955, .0318, .0390, .0458, .0586, .0187, .0236,
                .0285, .0384
            ]
        )
        betas = np.array(
            [
                0, .25, .5, 1, 0, .25, .5, 1, 0, .25, .5, 1, 0, .25, .5, 1, 0,
                .25, .5, 1
            ]
        )
        with np.errstate(all='ignore'), suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message="Density calculation unstable.*"
            )
            stats.levy_stable.pdf_default_method = method
            # stats.levy_stable.fft_grid_spacing = 0.0001
            # 计算稳定分布的概率密度函数，并断言其与已知密度值在指定精度下相等
            pdf = stats.levy_stable.pdf(xs, 1, betas, scale=1, loc=0)
            assert_almost_equal(
                pdf, density, decimal_places, method
            )

    @pytest.mark.parametrize(
        "params,expected",
        [
            [(1.48, -.22, 0, 1), (0, np.inf, np.nan, np.nan)],
            [(2, .9, 10, 1.5), (10, 4.5, 0, 0)]
        ]
    )
    def test_stats(self, params, expected):
        observed = stats.levy_stable.stats(
            params[0], params[1], loc=params[2], scale=params[3],
            moments='mvsk'
        )
        # 断言计算得到的统计量与预期值相等
        assert_almost_equal(observed, expected)
    @pytest.mark.parametrize('alpha', [0.25, 0.5, 0.75])
    @pytest.mark.parametrize(
        'function,beta,points,expected',
        [
            (
                stats.levy_stable.cdf,  # 使用 levy_stable 模块中的累积分布函数
                1.0,                    # 设定稳定分布参数 beta 为 1.0
                np.linspace(-25, 0, 10),# 生成从 -25 到 0 的等间隔数列作为测试点集合
                0.0,                    # 预期的累积分布函数值为 0.0
            ),
            (
                stats.levy_stable.pdf,  # 使用 levy_stable 模块中的概率密度函数
                1.0,                    # 设定稳定分布参数 beta 为 1.0
                np.linspace(-25, 0, 10),# 生成从 -25 到 0 的等间隔数列作为测试点集合
                0.0,                    # 预期的概率密度函数值为 0.0
            ),
            (
                stats.levy_stable.cdf,  # 使用 levy_stable 模块中的累积分布函数
                -1.0,                   # 设定稳定分布参数 beta 为 -1.0
                np.linspace(0, 25, 10), # 生成从 0 到 25 的等间隔数列作为测试点集合
                1.0,                    # 预期的累积分布函数值为 1.0
            ),
            (
                stats.levy_stable.pdf,  # 使用 levy_stable 模块中的概率密度函数
                -1.0,                   # 设定稳定分布参数 beta 为 -1.0
                np.linspace(0, 25, 10), # 生成从 0 到 25 的等间隔数列作为测试点集合
                0.0,                    # 预期的概率密度函数值为 0.0
            )
        ]
    )
    def test_distribution_outside_support(
            self, alpha, function, beta, points, expected
    ):
        """Ensure the pdf/cdf routines do not return nan outside support.

        This distribution's support becomes truncated in a few special cases:
            support is [mu, infty) if alpha < 1 and beta = 1
            support is (-infty, mu] if alpha < 1 and beta = -1
        Otherwise, the support is all reals. Here, mu is zero by default.
        """
        assert 0 < alpha < 1  # 断言 alpha 的值在 0 和 1 之间
        assert_almost_equal(   # 使用几乎相等的断言检查函数计算结果是否与预期值匹配
            function(points, alpha=alpha, beta=beta),  # 调用给定的函数计算在指定点集合上的函数值
            np.full(len(points), expected)             # 期望的结果与函数计算结果进行比较
        )

    @pytest.mark.parametrize(
        'x,alpha,beta,expected',
        # Reference values from Matlab
        # format long
        # alphas = [1.7720732804618808, 1.9217001522410235, 1.5654806051633634,
        #           1.7420803447784388, 1.5748002527689913];
        # betas = [0.5059373136902996, -0.8779442746685926, -0.4016220341911392,
        #          -0.38180029468259247, -0.25200194914153684];
        # x0s = [0, 1e-4, -1e-4];
        # for x0 = x0s
        #     disp("x0 = " + x0)
        #     for ii = 1:5
        #         alpha = alphas(ii);
        #         beta = betas(ii);
        #         pd = makedist('Stable','alpha',alpha,'beta',beta,'gam',1,'delta',0);
        #         % we need to adjust x. It is the same as x = 0 In scipy.
        #         x = x0 - beta * tan(pi * alpha / 2);
        #         disp(pd.pdf(x))
        #     end
        # end
        [
            (0, 1.7720732804618808, 0.5059373136902996, 0.278932636798268),
            (0, 1.9217001522410235, -0.8779442746685926, 0.281054757202316),
            (0, 1.5654806051633634, -0.4016220341911392, 0.271282133194204),
            (0, 1.7420803447784388, -0.38180029468259247, 0.280202199244247),
            (0, 1.5748002527689913, -0.25200194914153684, 0.280136576218665),
        ]
    )
    def test_x_equal_zeta(
            self, x, alpha, beta, expected
    ):
        # 测试 x 等于 ζ 的情况
        # 这里的测试用例是从 Matlab 中的参考值转换而来
        # 测试稳定分布的概率密度函数在不同参数下计算给定 x 值时的结果
    ):
        """
        Test pdf for x equal to zeta.

        With S1 parametrization: x0 = x + zeta if alpha != 1 So, for x = 0, x0
        will be close to zeta.

        When case "x equal zeta" is not handled properly and quad_eps is not
        low enough:
        - pdf may be less than 0
        - logpdf is nan

        The points from the parametrize block are found randomly so that PDF is
        less than 0.

        Reference values taken from MATLAB
        https://www.mathworks.com/help/stats/stable-distribution.html
        """
        # 设置稳定分布的数值积分精度
        stats.levy_stable.quad_eps = 1.2e-11

        # 断言稳定分布的概率密度函数在给定参数下接近期望值
        assert_almost_equal(
            stats.levy_stable.pdf(x, alpha=alpha, beta=beta),
            expected,
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        # 查看 test_x_equal_zeta 的注释以获取脚本的参考值
        'x,alpha,beta,expected',
        [
            (1e-4, 1.7720732804618808, 0.5059373136902996, 0.278929165340670),
            (1e-4, 1.9217001522410235, -0.8779442746685926, 0.281056564327953),
            (1e-4, 1.5654806051633634, -0.4016220341911392, 0.271252432161167),
            (1e-4, 1.7420803447784388, -0.38180029468259247, 0.280205311264134),
            (1e-4, 1.5748002527689913, -0.25200194914153684, 0.280140965235426),
            (-1e-4, 1.7720732804618808, 0.5059373136902996, 0.278936106741754),
            (-1e-4, 1.9217001522410235, -0.8779442746685926, 0.281052948629429),
            (-1e-4, 1.5654806051633634, -0.4016220341911392, 0.271275394392385),
            (-1e-4, 1.7420803447784388, -0.38180029468259247, 0.280199085645099),
            (-1e-4, 1.5748002527689913, -0.25200194914153684, 0.280132185432842),
        ]
    )
    def test_x_near_zeta(
            self, x, alpha, beta, expected
    ):
        """
        Test pdf for x near zeta.

        With S1 parametrization: x0 = x + zeta if alpha != 1 So, for x = 0, x0
        will be close to zeta.

        When case "x near zeta" is not handled properly and quad_eps is not
        low enough:
        - pdf may be less than 0
        - logpdf is nan

        The points from the parametrize block are found randomly so that PDF is
        less than 0.

        Reference values taken from MATLAB
        https://www.mathworks.com/help/stats/stable-distribution.html
        """
        # 设置稳定分布的数值积分精度
        stats.levy_stable.quad_eps = 1.2e-11

        # 断言稳定分布的概率密度函数在给定参数下接近期望值
        assert_almost_equal(
            stats.levy_stable.pdf(x, alpha=alpha, beta=beta),
            expected,
        )
class TestArrayArgument:  # test for ticket:992
    def setup_method(self):
        # 设置随机种子以确保可重复性
        np.random.seed(1234)

    def test_noexception(self):
        # 生成符合正态分布的随机数矩阵，指定均值为数组的索引，标准差为1
        rvs = stats.norm.rvs(loc=(np.arange(5)), scale=np.ones(5),
                             size=(10, 5))
        # 断言随机数矩阵的形状为(10, 5)
        assert_equal(rvs.shape, (10, 5))


class TestDocstring:
    def test_docstrings(self):
        # 检查正态分布函数的文档字符串中是否包含 "rayleigh"
        if stats.rayleigh.__doc__ is not None:
            assert_("rayleigh" in stats.rayleigh.__doc__.lower())
        # 检查伯努利分布函数的文档字符串中是否包含 "bernoulli"
        if stats.bernoulli.__doc__ is not None:
            assert_("bernoulli" in stats.bernoulli.__doc__.lower())

    def test_no_name_arg(self):
        # 如果未提供名称参数，应该能够构造而不出错。见 #1508。
        stats.rv_continuous()
        stats.rv_discrete()


def test_args_reduce():
    # 创建数组 a
    a = array([1, 3, 2, 1, 2, 3, 3])
    # 调用 argsreduce 函数，返回结果 b, c
    b, c = argsreduce(a > 1, a, 2)

    # 断言 b 的结果符合预期
    assert_array_equal(b, [3, 2, 2, 3, 3])
    # 断言 c 的结果符合预期
    assert_array_equal(c, [2])

    # 传入条件为真的情况，再次调用 argsreduce 函数，返回结果 b, c
    b, c = argsreduce(2 > 1, a, 2)
    # 断言 b 的结果符合预期
    assert_array_equal(b, a)
    # 断言 c 的结果符合预期
    assert_array_equal(c, [2] * np.size(a))

    # 传入条件为大于 0 的情况，再次调用 argsreduce 函数，返回结果 b, c
    b, c = argsreduce(a > 0, a, 2)
    # 断言 b 的结果符合预期
    assert_array_equal(b, a)
    # 断言 c 的结果符合预期
    assert_array_equal(c, [2] * np.size(a))


class TestFitMethod:
    # fitting assumes continuous parameters
    skip = ['ncf', 'ksone', 'kstwo', 'irwinhall']

    def setup_method(self):
        # 设置随机种子以确保可重复性
        np.random.seed(1234)

    # skip these b/c deprecated, or only loc and scale arguments
    fitSkipNonFinite = ['expon', 'norm', 'uniform', 'irwinhall']

    @pytest.mark.parametrize('dist,args', distcont)
    def test_fit_w_non_finite_data_values(self, dist, args):
        """gh-10300"""
        # 如果分布在 self.fitSkipNonFinite 中，跳过测试
        if dist in self.fitSkipNonFinite:
            pytest.skip("%s fit known to fail or deprecated" % dist)
        # 创建包含 NaN 和无穷值的数组 x, y
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        y = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        # 获取分布函数对象
        distfunc = getattr(stats, dist)
        # 断言分布拟合函数在给定非有限数据时会引发 ValueError 异常
        assert_raises(ValueError, distfunc.fit, x, fscale=1)
        assert_raises(ValueError, distfunc.fit, y, fscale=1)

    def test_fix_fit_2args_lognorm(self):
        # 回归测试 #1551
        np.random.seed(12345)
        with np.errstate(all='ignore'):
            # 生成 lognorm 分布的随机变量数组 x
            x = stats.lognorm.rvs(0.25, 0., 20.0, size=20)
            # 期望的形状值
            expected_shape = np.sqrt(((np.log(x) - np.log(20))**2).mean())
            # 断言 lognorm 分布的拟合结果与期望值接近
            assert_allclose(np.array(stats.lognorm.fit(x, floc=0, fscale=20)),
                            [expected_shape, 0, 20], atol=1e-8)

    def test_fix_fit_norm(self):
        # 创建数组 x
        x = np.arange(1, 6)

        # 拟合正态分布，返回拟合结果的均值 loc 和标准差 scale
        loc, scale = stats.norm.fit(x)
        # 断言拟合结果的均值 loc 等于预期值 3
        assert_almost_equal(loc, 3)
        # 断言拟合结果的标准差 scale 等于预期值 sqrt(2)
        assert_almost_equal(scale, np.sqrt(2))

        # 拟合正态分布，指定均值的先验值为 2，返回拟合结果的均值 loc 和标准差 scale
        loc, scale = stats.norm.fit(x, floc=2)
        # 断言拟合结果的均值 loc 等于预期值 2
        assert_equal(loc, 2)
        # 断言拟合结果的标准差 scale 等于预期值 sqrt(3)
        assert_equal(scale, np.sqrt(3))

        # 拟合正态分布，指定标准差的先验值为 2，返回拟合结果的均值 loc 和标准差 scale
        loc, scale = stats.norm.fit(x, fscale=2)
        # 断言拟合结果的均值 loc 等于预期值 3
        assert_almost_equal(loc, 3)
        # 断言拟合结果的标准差 scale 等于预期值 2
        assert_equal(scale, 2)
    # 定义一个测试方法，用于测试 gamma 分布拟合参数时的不同情况
    def test_fix_fit_gamma(self):
        # 创建一个包含从1到5的数组
        x = np.arange(1, 6)
        # 计算数组元素的自然对数的平均值
        meanlog = np.log(x).mean()

        # 使用 floc=0 进行 gamma 分布拟合的基本测试
        floc = 0
        # 调用 gamma.fit 函数进行拟合，返回参数 a, loc, scale
        a, loc, scale = stats.gamma.fit(x, floc=floc)
        # 计算预期的数学公式结果
        s = np.log(x.mean()) - meanlog
        # 断言 gamma 分布参数的对数形式与特定公式的差值
        assert_almost_equal(np.log(a) - special.digamma(a), s, decimal=5)
        # 断言 loc 参数与预期的 floc 值相等
        assert_equal(loc, floc)
        # 断言 scale 参数与 x 均值除以 a 的值相等
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        # Regression tests for gh-2514.
        # 用于测试修复 gh-2514 的回归测试
        # 问题在于如果给定了 `floc=0`，则任何其他固定参数都会被忽略。

        # 测试当指定 f0=1 和 floc=0 时的情况
        f0 = 1
        floc = 0
        # 调用 gamma.fit 函数进行拟合，返回参数 a, loc, scale
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        # 断言 a 参数与 f0 相等
        assert_equal(a, f0)
        # 断言 loc 参数与 floc 相等
        assert_equal(loc, floc)
        # 断言 scale 参数与 x 均值除以 a 的值相等
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        # 测试当指定 f0=2 和 floc=0 时的情况
        f0 = 2
        floc = 0
        # 调用 gamma.fit 函数进行拟合，返回参数 a, loc, scale
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        # 断言 a 参数与 f0 相等
        assert_equal(a, f0)
        # 断言 loc 参数与 floc 相等
        assert_equal(loc, floc)
        # 断言 scale 参数与 x 均值除以 a 的值相等
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        # 测试当指定 floc=0 和 fscale=2 时的情况，loc 和 scale 固定
        floc = 0
        fscale = 2
        # 调用 gamma.fit 函数进行拟合，返回参数 a, loc, scale
        a, loc, scale = stats.gamma.fit(x, floc=floc, fscale=fscale)
        # 断言 loc 参数与 floc 相等
        assert_equal(loc, floc)
        # 断言 scale 参数与 fscale 相等
        assert_equal(scale, fscale)
        # 计算预期的 c 值
        c = meanlog - np.log(fscale)
        # 断言特殊函数 digamma(a) 的结果与预期的 c 值相等
        assert_almost_equal(special.digamma(a), c)
    def test_fix_fit_beta(self):
        # Test beta.fit when both floc and fscale are given.

        def mlefunc(a, b, x):
            # 定义最大似然函数的关键点是该函数的零点。
            n = len(x)
            s1 = np.log(x).sum()
            s2 = np.log(1-x).sum()
            psiab = special.psi(a + b)
            # 计算最大似然函数的两个部分
            func = [s1 - n * (-psiab + special.psi(a)),
                    s2 - n * (-psiab + special.psi(b))]
            return func

        # 基本测试，给定 floc 和 fscale
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, floc=0, fscale=1)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        assert_allclose(mlefunc(a, b, x), [0, 0], atol=1e-6)

        # 基本测试，给定 f0, floc 和 fscale
        # 这也是 gh-2514 的回归测试
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, f0=2, floc=0, fscale=1)
        assert_equal(a, 2)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        da, db = mlefunc(a, b, x)
        assert_allclose(db, 0, atol=1e-5)

        # 使用相同的 floc 和 fscale 值，但是反转数据并固定 b (f1)
        x2 = 1 - x
        a2, b2, loc2, scale2 = stats.beta.fit(x2, f1=2, floc=0, fscale=1)
        assert_equal(b2, 2)
        assert_equal(loc2, 0)
        assert_equal(scale2, 1)
        da, db = mlefunc(a2, b2, x2)
        assert_allclose(da, 0, atol=1e-5)
        # 这个测试中的 a2 应该等于上面的 b。
        assert_almost_equal(a2, b)

        # 检查当给定 floc 和 fscale 时，数据超出边界的检测
        assert_raises(ValueError, stats.beta.fit, x, floc=0.5, fscale=1)
        y = np.array([0, .5, 1])
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f0=2)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f1=2)

        # 检查尝试固定所有参数时是否引发 ValueError
        assert_raises(ValueError, stats.beta.fit, y, f0=0, f1=1,
                      floc=2, fscale=3)

    def test_expon_fit(self):
        x = np.array([2, 2, 4, 4, 4, 4, 4, 8])

        loc, scale = stats.expon.fit(x)
        assert_equal(loc, 2)    # x.min()
        assert_equal(scale, 2)  # x.mean() - x.min()

        loc, scale = stats.expon.fit(x, fscale=3)
        assert_equal(loc, 2)    # x.min()
        assert_equal(scale, 3)  # fscale

        loc, scale = stats.expon.fit(x, floc=0)
        assert_equal(loc, 0)    # floc
        assert_equal(scale, 4)  # x.mean() - loc
    # 定义一个测试函数，用于测试对数正态分布的拟合
    def test_lognorm_fit(self):
        # 定义测试数据 x，这里是一个 NumPy 数组
        x = np.array([1.5, 3, 10, 15, 23, 59])
        # 计算 x-1 的自然对数
        lnxm1 = np.log(x - 1)

        # 使用对数正态分布拟合 x 数据，返回形状、位置和尺度参数
        shape, loc, scale = stats.lognorm.fit(x, floc=1)
        # 断言形状参数近似等于 lnxm1 的标准差
        assert_allclose(shape, lnxm1.std(), rtol=1e-12)
        # 断言位置参数等于 1
        assert_equal(loc, 1)
        # 断言尺度参数近似等于 lnxm1 的均值的指数函数
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)

        # 使用对数正态分布拟合 x 数据，指定位置为 1，尺度为 6
        shape, loc, scale = stats.lognorm.fit(x, floc=1, fscale=6)
        # 断言形状参数近似等于 lnxm1 减去 ln(6) 后的平均值的平方根
        assert_allclose(shape, np.sqrt(((lnxm1 - np.log(6))**2).mean()),
                        rtol=1e-12)
        # 断言位置参数等于 1
        assert_equal(loc, 1)
        # 断言尺度参数等于 6
        assert_equal(scale, 6)

        # 使用对数正态分布拟合 x 数据，指定位置为 1，固定形状为 0.75
        shape, loc, scale = stats.lognorm.fit(x, floc=1, fix_s=0.75)
        # 断言形状参数等于 0.75
        assert_equal(shape, 0.75)
        # 断言位置参数等于 1
        assert_equal(loc, 1)
        # 断言尺度参数近似等于 lnxm1 的均值的指数函数
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)

    # 定义一个测试函数，用于测试均匀分布的拟合
    def test_uniform_fit(self):
        # 定义测试数据 x，这里是一个 NumPy 数组
        x = np.array([1.0, 1.1, 1.2, 9.0])

        # 使用均匀分布拟合 x 数据，返回位置和尺度参数
        loc, scale = stats.uniform.fit(x)
        # 断言位置参数等于 x 中的最小值
        assert_equal(loc, x.min())
        # 断言尺度参数等于 x 中的极差
        assert_equal(scale, np.ptp(x))

        # 使用均匀分布拟合 x 数据，指定位置为 0
        loc, scale = stats.uniform.fit(x, floc=0)
        # 断言位置参数等于 0
        assert_equal(loc, 0)
        # 断言尺度参数等于 x 中的最大值
        assert_equal(scale, x.max())

        # 使用均匀分布拟合 x 数据，指定尺度为 10
        loc, scale = stats.uniform.fit(x, fscale=10)
        # 断言位置参数等于 0
        assert_equal(loc, 0)
        # 断言尺度参数等于 10
        assert_equal(scale, 10)

        # 断言拟合过程中会引发 ValueError 异常，指定位置参数为 2.0
        assert_raises(ValueError, stats.uniform.fit, x, floc=2.0)
        # 断言拟合过程中会引发 ValueError 异常，指定尺度参数为 5.0
        assert_raises(ValueError, stats.uniform.fit, x, fscale=5.0)
    # 定义一个测试方法，用于验证 beta 分布的拟合情况
    def test_fshapes(self, method):
        # 设置 beta 分布的参数 a 和 b
        a, b = 3., 4.
        # 从 beta 分布中生成随机样本数据 x
        x = stats.beta.rvs(a, b, size=100, random_state=1234)
        
        # 使用 beta 分布的拟合函数进行拟合，通过位置参数 f0
        res_1 = stats.beta.fit(x, f0=3., method=method)
        # 使用命名参数 fa 进行拟合，应当等价于 f0
        res_2 = stats.beta.fit(x, fa=3., method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)

        # 使用命名参数 fix_a 进行拟合，应当等价于 f0
        res_2 = stats.beta.fit(x, fix_a=3., method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)

        # 使用 beta 分布的拟合函数进行拟合，通过位置参数 f1
        res_3 = stats.beta.fit(x, f1=4., method=method)
        # 使用命名参数 fb 进行拟合，应当等价于 f1
        res_4 = stats.beta.fit(x, fb=4., method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)

        # 使用命名参数 fix_b 进行拟合，应当等价于 f1
        res_4 = stats.beta.fit(x, fix_b=4., method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)

        # 尝试同时指定位置参数和命名参数，应当引发 ValueError 异常
        assert_raises(ValueError, stats.beta.fit, x, fa=1, f0=2, method=method)

        # 检查指定所有参数固定时是否会引发 ValueError 异常
        assert_raises(ValueError, stats.beta.fit, x, fa=0, f1=1,
                      floc=2, fscale=3, method=method)

        # 对 beta 分布和 gamma 分布的拟合方法进行参数检查
        # beta 分布指定 fa, floc, fscale，应当返回对应的参数值
        res_5 = stats.beta.fit(x, fa=3., floc=0, fscale=1, method=method)
        aa, bb, ll, ss = res_5
        assert_equal([aa, ll, ss], [3., 0, 1])

        # gamma 分布的拟合方法，指定 fa 参数应当返回相同的参数值
        a = 3.
        data = stats.gamma.rvs(a, size=100)
        aa, ll, ss = stats.gamma.fit(data, fa=a, method=method)
        assert_equal(aa, a)

    # 使用 pytest 的参数化装饰器定义测试方法，参数为拟合方法 MLE 和 MM
    @pytest.mark.parametrize("method", ["MLE", "MM"])
    def test_extra_params(self, method):
        # 对于未知参数应当引发 TypeError 异常，而不是被静默忽略
        dist = stats.exponnorm
        data = dist.rvs(K=2, size=100)
        # 尝试使用未知参数 dct 进行拟合，应当引发 TypeError 异常
        dct = dict(enikibeniki=-101)
        assert_raises(TypeError, dist.fit, data, **dct, method=method)
class TestFrozen:
    # 设置测试方法的初始化，确保随机种子固定为1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试冻结分布与原始对象给出相同结果的情况
    #
    # 只针对正态分布（指定 loc 和 scale）和伽马分布（指定形状参数）进行测试
    def test_norm(self):
        # 使用 scipy.stats 中的正态分布对象作为比较标准
        dist = stats.norm
        # 创建一个冻结的正态分布对象，指定 loc=10.0, scale=3.0
        frozen = stats.norm(loc=10.0, scale=3.0)

        # 测试概率密度函数 (PDF) 的一致性
        result_f = frozen.pdf(20.0)
        result = dist.pdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试累积分布函数 (CDF) 的一致性
        result_f = frozen.cdf(20.0)
        result = dist.cdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试累积分布函数的反函数 (PPF) 的一致性
        result_f = frozen.ppf(0.25)
        result = dist.ppf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试累积分布函数的逆函数 (ISF) 的一致性
        result_f = frozen.isf(0.25)
        result = dist.isf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试生存函数 (SF) 的一致性
        result_f = frozen.sf(10.0)
        result = dist.sf(10.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试中位数的一致性
        result_f = frozen.median()
        result = dist.median(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试均值的一致性
        result_f = frozen.mean()
        result = dist.mean(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试方差的一致性
        result_f = frozen.var()
        result = dist.var(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试标准差的一致性
        result_f = frozen.std()
        result = dist.std(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试熵的一致性
        result_f = frozen.entropy()
        result = dist.entropy(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 测试矩的一致性（这里是二阶矩）
        result_f = frozen.moment(2)
        result = dist.moment(2, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        # 检查冻结分布对象的下界（a）是否与原始分布对象的下界（a）一致
        assert_equal(frozen.a, dist.a)
        # 检查冻结分布对象的上界（b）是否与原始分布对象的上界（b）一致
        assert_equal(frozen.b, dist.b)
    def test_gamma(self):
        # 设置参数 a 为 2.0
        a = 2.0
        # 使用 scipy.stats 模块中的 gamma 分布函数对象
        dist = stats.gamma
        # 创建一个冻结的 gamma 分布对象，指定参数 a
        frozen = stats.gamma(a)

        # 计算冻结对象的概率密度函数在 x=20.0 处的取值
        result_f = frozen.pdf(20.0)
        # 计算未冻结的 gamma 分布的概率密度函数在 x=20.0 处的取值，参数为 a
        result = dist.pdf(20.0, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的累积分布函数在 x=20.0 处的取值
        result_f = frozen.cdf(20.0)
        # 计算未冻结的 gamma 分布的累积分布函数在 x=20.0 处的取值，参数为 a
        result = dist.cdf(20.0, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的累积分布函数的反函数在 p=0.25 处的取值
        result_f = frozen.ppf(0.25)
        # 计算未冻结的 gamma 分布的累积分布函数的反函数在 p=0.25 处的取值，参数为 a
        result = dist.ppf(0.25, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的累积分布函数的反函数在 p=0.25 处的取值
        result_f = frozen.isf(0.25)
        # 计算未冻结的 gamma 分布的累积分布函数的反函数在 p=0.25 处的取值，参数为 a
        result = dist.isf(0.25, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的生存函数在 x=10.0 处的取值
        result_f = frozen.sf(10.0)
        # 计算未冻结的 gamma 分布的生存函数在 x=10.0 处的取值，参数为 a
        result = dist.sf(10.0, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的中位数
        result_f = frozen.median()
        # 计算未冻结的 gamma 分布的中位数，参数为 a
        result = dist.median(a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的期望值（均值）
        result_f = frozen.mean()
        # 计算未冻结的 gamma 分布的期望值（均值），参数为 a
        result = dist.mean(a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的方差
        result_f = frozen.var()
        # 计算未冻结的 gamma 分布的方差，参数为 a
        result = dist.var(a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的标准差
        result_f = frozen.std()
        # 计算未冻结的 gamma 分布的标准差，参数为 a
        result = dist.std(a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的熵
        result_f = frozen.entropy()
        # 计算未冻结的 gamma 分布的熵，参数为 a
        result = dist.entropy(a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 计算冻结对象的矩（2阶）
        result_f = frozen.moment(2)
        # 计算未冻结的 gamma 分布的矩（2阶），参数为 a
        result = dist.moment(2, a)
        # 断言两者结果相等
        assert_equal(result_f, result)

        # 断言冻结对象的参数 a 与未冻结对象的参数 a 相等
        assert_equal(frozen.a, frozen.dist.a)
        # 断言冻结对象的参数 b 与未冻结对象的参数 b 相等
        assert_equal(frozen.b, frozen.dist.b)

    def test_regression_ticket_1293(self):
        # 创建一个参数为 1 的 lognorm 分布的冻结对象
        frozen = stats.lognorm(1)
        # 调用冻结对象的 moment 方法计算2阶矩
        m1 = frozen.moment(2)
        # 调用 stats 方法，传入 moments='mvsk' 参数，无返回值
        frozen.stats(moments='mvsk')
        # 再次调用冻结对象的 moment 方法计算2阶矩
        m2 = frozen.moment(2)
        # 断言两次计算出的2阶矩结果相等
        assert_equal(m1, m2)
    def test_ab(self):
        # 测试冻结分布的支持
        # (i) 即使对原始分布进行更改，其支持仍然保持冻结状态
        # (ii) 如果形状参数使得[a, b]的值不是默认值[0, inf]，则实际上是正确的
        # 以广义帕累托分布为例，其支持取决于形状参数的值：
        # 对于 c > 0: a, b = 0, inf
        # 对于 c < 0: a, b = 0, -1/c

        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0., 10.])

        c = 0.1
        stats.genpareto.pdf(0, c=c)
        assert_equal(rv.dist._get_support(c), [0, np.inf])

        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0., 10.])

        c = 0.1
        stats.genpareto.pdf(0, c)  # 这不应该改变 genpareto.b
        assert_equal((rv.dist.a, rv.dist.b), stats.genpareto._get_support(c))

        rv1 = stats.genpareto(c=0.1)
        assert_(rv1.dist is not rv.dist)

        # c >= 0: a, b = [0, inf]
        for c in [1., 0.]:
            c = np.asarray(c)
            rv = stats.genpareto(c=c)
            a, b = rv.a, rv.b
            assert_equal(a, 0.)
            assert_(np.isposinf(b))

            # c < 0: a=0, b=1/|c|
            c = np.asarray(-2.)
            a, b = stats.genpareto._get_support(c)
            assert_allclose([a, b], [0., 0.5])

    def test_rv_frozen_in_namespace(self):
        # gh-3522 的回归测试
        assert_(hasattr(stats.distributions, 'rv_frozen'))

    def test_random_state(self):
        # 仅检查 random_state 属性是否存在
        frozen = stats.norm()
        assert_(hasattr(frozen, 'random_state'))

        # ... 确保可以设置 random_state 属性
        frozen.random_state = 42
        assert_equal(frozen.random_state.get_state(),
                     np.random.RandomState(42).get_state())

        # ... 并且 .rvs 方法能够接受它作为参数
        rndm = np.random.RandomState(1234)
        frozen.rvs(size=8, random_state=rndm)
    def test_pickling(self):
        # 测试一个冻结的实例是否可以序列化和反序列化
        # (该方法是 common_tests.check_pickling 的一个克隆)
        
        # 创建一个 Beta 分布实例
        beta = stats.beta(2.3098496451481823, 0.62687954300963677)
        # 创建一个泊松分布实例
        poiss = stats.poisson(3.)
        # 创建一个离散随机变量实例
        sample = stats.rv_discrete(values=([0, 1, 2, 3],
                                           [0.1, 0.2, 0.3, 0.4]))

        # 对每个分布实例进行以下操作
        for distfn in [beta, poiss, sample]:
            # 设置随机数生成器的种子
            distfn.random_state = 1234
            # 生成随机变量
            distfn.rvs(size=8)
            # 序列化实例对象
            s = pickle.dumps(distfn)
            # 生成新的随机变量
            r0 = distfn.rvs(size=8)

            # 反序列化对象
            unpickled = pickle.loads(s)
            # 生成反序列化后的随机变量
            r1 = unpickled.rvs(size=8)
            # 断言序列化前后生成的随机变量是否一致
            assert_equal(r0, r1)

            # 进行一些方法的基本测试
            # 测试中位数的一致性
            medians = [distfn.ppf(0.5), unpickled.ppf(0.5)]
            assert_equal(medians[0], medians[1])
            # 测试累积分布函数的一致性
            assert_equal(distfn.cdf(medians[0]),
                         unpickled.cdf(medians[1]))

    def test_expect(self):
        # 对冻结分布的 expect 方法进行基本测试
        # 只针对具有 loc 和 scale 参数的 Gamma 分布以及具有指定 loc 参数的泊松分布

        # 定义一个简单的函数
        def func(x):
            return x

        # 创建 Gamma 分布实例
        gm = stats.gamma(a=2, loc=3, scale=4)
        # 在特定的错误状态下计算 expect 方法的值
        with np.errstate(invalid="ignore", divide="ignore"):
            gm_val = gm.expect(func, lb=1, ub=2, conditional=True)
            # 调用 scipy.stats.gamma 的 expect 方法
            gamma_val = stats.gamma.expect(func, args=(2,), loc=3, scale=4,
                                           lb=1, ub=2, conditional=True)
        # 使用数值比较函数检查两个值的近似程度
        assert_allclose(gm_val, gamma_val)

        # 创建泊松分布实例
        p = stats.poisson(3, loc=4)
        # 调用泊松分布的 expect 方法
        p_val = p.expect(func)
        # 调用 scipy.stats.poisson 的 expect 方法
        poisson_val = stats.poisson.expect(func, args=(3,), loc=4)
        # 使用数值比较函数检查两个值的近似程度
        assert_allclose(p_val, poisson_val)
class TestExpect:
    # Test for expect method.
    #
    # Uses normal distribution and beta distribution for finite bounds, and
    # hypergeom for discrete distribution with finite support

    # 对正态分布进行测试
    def test_norm(self):
        # 计算正态分布期望值，lambda 函数为 (x-5)*(x-5)，期望值的位置 loc=5，标准差 scale=2
        v = stats.norm.expect(lambda x: (x-5)*(x-5), loc=5, scale=2)
        # 断言期望值 v 约等于 4，精确到小数点后 14 位
        assert_almost_equal(v, 4, decimal=14)

        # 计算正态分布期望值，lambda 函数为 x，期望值的位置 loc=5，标准差 scale=2
        m = stats.norm.expect(lambda x: (x), loc=5, scale=2)
        # 断言期望值 m 约等于 5，精确到小数点后 14 位
        assert_almost_equal(m, 5, decimal=14)

        # 计算正态分布的累积分布函数的逆函数，确定置信区间的下界 lb 和上界 ub
        lb = stats.norm.ppf(0.05, loc=5, scale=2)
        ub = stats.norm.ppf(0.95, loc=5, scale=2)
        # 计算正态分布中落在置信区间 [lb, ub] 内的概率，lambda 函数为常数函数 1
        prob90 = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub)
        # 断言概率 prob90 约等于 0.9，精确到小数点后 14 位
        assert_almost_equal(prob90, 0.9, decimal=14)

        # 计算正态分布中落在置信区间 [lb, ub] 内的概率，lambda 函数为常数函数 1，条件为 True
        prob90c = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub,
                                    conditional=True)
        # 断言条件概率 prob90c 约等于 1，精确到小数点后 14 位
        assert_almost_equal(prob90c, 1., decimal=14)

    # 对 beta 分布进行测试
    def test_beta(self):
        # 计算 beta 分布期望值，lambda 函数为 (x-19/3.)*(x-19/3.)，参数为 (10, 5)，期望值的位置 loc=5，标准差 scale=2
        v = stats.beta.expect(lambda x: (x-19/3.)*(x-19/3.), args=(10, 5),
                              loc=5, scale=2)
        # 断言期望值 v 约等于 1/18，精确到小数点后 13 位
        assert_almost_equal(v, 1./18., decimal=13)

        # 计算 beta 分布期望值，lambda 函数为 x，参数为 (10, 5)，期望值的位置 loc=5，标准差 scale=2
        m = stats.beta.expect(lambda x: x, args=(10, 5), loc=5., scale=2.)
        # 断言期望值 m 约等于 19/3，精确到小数点后 13 位
        assert_almost_equal(m, 19/3., decimal=13)

        # 计算 beta 分布的累积分布函数的逆函数，确定置信区间的下界 lb 和上界 ub
        ub = stats.beta.ppf(0.95, 10, 10, loc=5, scale=2)
        lb = stats.beta.ppf(0.05, 10, 10, loc=5, scale=2)
        # 计算 beta 分布中落在置信区间 [lb, ub] 内的概率，lambda 函数为常数函数 1
        prob90 = stats.beta.expect(lambda x: 1., args=(10, 10), loc=5.,
                                   scale=2., lb=lb, ub=ub, conditional=False)
        # 断言概率 prob90 约等于 0.9，精确到小数点后 13 位
        assert_almost_equal(prob90, 0.9, decimal=13)

        # 计算 beta 分布中落在置信区间 [lb, ub] 内的概率，lambda 函数为常数函数 1，条件为 True
        prob90c = stats.beta.expect(lambda x: 1, args=(10, 10), loc=5,
                                    scale=2, lb=lb, ub=ub, conditional=True)
        # 断言条件概率 prob90c 约等于 1，精确到小数点后 13 位
        assert_almost_equal(prob90c, 1., decimal=13)
    def test_hypergeom(self):
        # hypergeom 模块的单元测试

        # 未指定边界的测试用例
        m_true, v_true = stats.hypergeom.stats(20, 10, 8, loc=5.)
        # 计算超几何分布的期望，使用均值作为位置参数
        m = stats.hypergeom.expect(lambda x: x, args=(20, 10, 8), loc=5.)
        # 断言期望值与真实值近似相等，精度到小数点后13位
        assert_almost_equal(m, m_true, decimal=13)

        # 计算方差的期望，以 (x-9.)^2 作为函数，使用均值作为位置参数
        v = stats.hypergeom.expect(lambda x: (x-9.)**2, args=(20, 10, 8),
                                   loc=5.)
        # 断言方差的期望与真实值近似相等，精度到小数点后14位
        assert_almost_equal(v, v_true, decimal=14)

        # 带有边界的测试用例，边界与平移后的支持区域相等
        v_bounds = stats.hypergeom.expect(lambda x: (x-9.)**2,
                                          args=(20, 10, 8),
                                          loc=5., lb=5, ub=13)
        # 断言带边界的方差的期望与真实值近似相等，精度到小数点后14位
        assert_almost_equal(v_bounds, v_true, decimal=14)

        # 排除边界点的概率计算
        prob_true = 1-stats.hypergeom.pmf([5, 13], 20, 10, 8, loc=5).sum()
        # 计算概率的期望，使用常数函数 1，使用均值作为位置参数，设定下界和上界
        prob_bounds = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8),
                                             loc=5., lb=6, ub=12)
        # 断言条件下的概率的期望与真实值近似相等，精度到小数点后13位
        assert_almost_equal(prob_bounds, prob_true, decimal=13)

        # 条件概率计算
        prob_bc = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.,
                                         lb=6, ub=12, conditional=True)
        # 断言条件下的概率的期望与真实值近似相等，精度到小数点后14位
        assert_almost_equal(prob_bc, 1, decimal=14)

        # 检查简单积分
        prob_b = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8),
                                        lb=0, ub=8)
        # 断言简单积分的期望与真实值近似相等，精度到小数点后13位
        assert_almost_equal(prob_b, 1, decimal=13)

    def test_poisson(self):
        # poisson 模块的单元测试，仅使用下界

        # 使用边界计算概率的期望，使用常数函数 1
        prob_bounds = stats.poisson.expect(lambda x: 1, args=(2,), lb=3,
                                           conditional=False)
        # 计算真实的概率的期望，使用累积分布函数
        prob_b_true = 1-stats.poisson.cdf(2, 2)
        # 断言边界计算的概率的期望与真实值近似相等，精度到小数点后14位
        assert_almost_equal(prob_bounds, prob_b_true, decimal=14)

        # 使用条件下的边界计算概率的期望，使用常数函数 1
        prob_lb = stats.poisson.expect(lambda x: 1, args=(2,), lb=2,
                                       conditional=True)
        # 断言条件下的边界计算的概率的期望与真实值近似相等，精度到小数点后14位
        assert_almost_equal(prob_lb, 1, decimal=14)

    def test_genhalflogistic(self):
        # genhalflogistic 模块的单元测试，更改支持的上界在 _argcheck 中

        # 回归测试 gh-2622，检查两次使用相同输入调用 expect 的一致性
        halflog = stats.genhalflogistic
        res1 = halflog.expect(args=(1.5,))
        halflog.expect(args=(0.5,))
        res2 = halflog.expect(args=(1.5,))
        # 断言两次调用 expect 得到的结果近似相等，精度到小数点后14位
        assert_almost_equal(res1, res2, decimal=14)

    def test_rice_overflow(self):
        # rice.pdf(999, 0.74) 由于 special.i0 溢出而为无穷
        # 检查使用 i0e 是否修复了问题

        # 断言 rice.pdf(999, 0.74) 的结果是有限的
        assert_(np.isfinite(stats.rice.pdf(999, 0.74)))

        # 断言使用常数函数 1 计算的期望是有限的
        assert_(np.isfinite(stats.rice.expect(lambda x: 1, args=(0.74,))))
        # 断言使用常数函数 2 计算的期望是有限的
        assert_(np.isfinite(stats.rice.expect(lambda x: 2, args=(0.74,))))
        # 断言使用常数函数 3 计算的期望是有限的
        assert_(np.isfinite(stats.rice.expect(lambda x: 3, args=(0.74,))))
    def test_logser(self):
        # 测试具有无限支持和位置参数的离散分布
        p, loc = 0.3, 3
        # 计算期望值，使用 stats.logser.expect 函数，参数为 lambda 函数和参数 p
        res_0 = stats.logser.expect(lambda k: k, args=(p,))
        # 检查结果是否与正确答案接近（几何级数的和）
        assert_allclose(res_0,
                        p / (p - 1.) / np.log(1. - p), atol=1e-15)

        # 现在检查带有 `loc` 参数的情况
        # 使用 stats.logser.expect 函数，参数为 lambda 函数、参数 p 和 loc
        res_l = stats.logser.expect(lambda k: k, args=(p,), loc=loc)
        assert_allclose(res_l, res_0 + loc, atol=1e-15)

    def test_skellam(self):
        # 使用具有双向无限支持的离散分布。计算前两个矩并与已知值比较（见 skellam.stats）
        p1, p2 = 18, 22
        # 计算期望值，使用 stats.skellam.expect 函数，参数为 lambda 函数和参数 p1, p2
        m1 = stats.skellam.expect(lambda x: x, args=(p1, p2))
        # 计算二阶中心矩，使用 stats.skellam.expect 函数，参数为 lambda 函数和参数 p1, p2
        m2 = stats.skellam.expect(lambda x: x**2, args=(p1, p2))
        assert_allclose(m1, p1 - p2, atol=1e-12)
        assert_allclose(m2 - m1**2, p1 + p2, atol=1e-12)

    def test_randint(self):
        # 使用具有参数相关支持的离散分布，支持范围大于默认的块大小
        lo, hi = 0, 113
        # 计算期望值，使用 stats.randint.expect 函数，参数为 lambda 函数和参数范围 (lo, hi)
        res = stats.randint.expect(lambda x: x, (lo, hi))
        assert_allclose(res,
                        sum(_ for _ in range(lo, hi)) / (hi - lo), atol=1e-15)

    def test_zipf(self):
        # 测试即使总和发散，也不会出现无限循环的情况
        # 断言会产生 RuntimeWarning，使用 stats.zipf.expect 函数，参数为 lambda 函数和参数 (2,)
        assert_warns(RuntimeWarning, stats.zipf.expect,
                     lambda x: x**2, (2,))

    def test_discrete_kwds(self):
        # 检查离散分布的 expect 函数是否接受关键字来控制求和过程
        n0 = stats.poisson.expect(lambda x: 1, args=(2,))
        n1 = stats.poisson.expect(lambda x: 1, args=(2,),
                                  maxcount=1001, chunksize=32, tolerance=1e-8)
        assert_almost_equal(n0, n1, decimal=14)

    def test_moment(self):
        # 测试 .moment() 方法：计算更高阶矩并与已知值比较
        def poiss_moment5(mu):
            return mu**5 + 10*mu**4 + 25*mu**3 + 15*mu**2 + mu

        for mu in [5, 7]:
            # 计算第五阶矩，使用 stats.poisson.moment 方法，参数为 5 和参数 mu
            m5 = stats.poisson.moment(5, mu)
            assert_allclose(m5, poiss_moment5(mu), rtol=1e-10)

    def test_challenging_cases_gh8928(self):
        # 在 gh-8928 中报告了 expect 函数未能产生正确结果的几种情况。检查这些情况是否已解决。
        assert_allclose(stats.norm.expect(loc=36, scale=1.0), 36)
        assert_allclose(stats.norm.expect(loc=40, scale=1.0), 40)
        assert_allclose(stats.norm.expect(loc=10, scale=0.1), 10)
        assert_allclose(stats.gamma.expect(args=(148,)), 148)
        assert_allclose(stats.logistic.expect(loc=85), 85)
    # 定义一个测试方法，用于测试 `gh15855` 中对 `expect` 方法的改动是否正确处理 lb（下限）和 ub（上限）
    def test_lb_ub_gh15855(self):
        # 使用均匀分布作为测试分布
        dist = stats.uniform
        # 计算基准期望，位置参数为 10，尺度参数为 5，即期望值为 12.5
        ref = dist.mean(loc=10, scale=5)  # 12.5
        # 计算整个分布上的矩，应该等于基准期望
        assert_allclose(dist.expect(loc=10, scale=5), ref)
        # 在整个分布上计算矩，lb 和 ub 超出支持范围
        assert_allclose(dist.expect(loc=10, scale=5, lb=9, ub=16), ref)
        # 在分布的 60% 范围内计算矩，[lb, ub] 范围在支持范围内并居中
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14), ref*0.6)
        # 在截断分布上计算矩，实质上是在 lb 和 ub 之间的条件下计算
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14,
                                    conditional=True), ref)
        # 在分布的 40% 范围内计算矩，[lb, ub] 范围不在支持范围内并不居中
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=13), 12*0.4)
        # lb 大于 ub 的情况下计算矩
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11), -12*0.4)
        # lb 大于 ub 的情况下，在条件为真时计算矩
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11,
                                    conditional=True), 12)
class TestNct:
    def test_nc_parameter(self):
        # Parameter values c<=0 were not enabled (gh-2402).
        # For negative values c and for c=0 results of rv.cdf(0) below were nan
        # 创建一个非中心 t 分布的随机变量 rv，其中 df=5, nc=0
        rv = stats.nct(5, 0)
        # 断言 rv.cdf(0) 的计算结果为 0.5
        assert_equal(rv.cdf(0), 0.5)
        # 创建另一个非中心 t 分布的随机变量 rv，其中 df=5, nc=-1
        rv = stats.nct(5, -1)
        # 断言 rv.cdf(0) 的计算结果接近于 0.841344746069，精确度为 10 位小数
        assert_almost_equal(rv.cdf(0), 0.841344746069, decimal=10)

    def test_broadcasting(self):
        # 对于给定的参数，计算非中心 t 分布的概率密度函数
        res = stats.nct.pdf(5, np.arange(4, 7)[:, None],
                            np.linspace(0.1, 1, 4))
        # 预期的结果数组
        expected = array([[0.00321886, 0.00557466, 0.00918418, 0.01442997],
                          [0.00217142, 0.00395366, 0.00683888, 0.01126276],
                          [0.00153078, 0.00291093, 0.00525206, 0.00900815]])
        # 断言 res 数组与 expected 数组非常接近，相对误差容差为 1e-5
        assert_allclose(res, expected, rtol=1e-5)

    def test_variance_gh_issue_2401(self):
        # 计算 df=4, nc=0 的非中心 t 分布的随机变量 rv
        rv = stats.nct(4, 0)
        # 断言 rv.var() 的计算结果为 2.0
        assert_equal(rv.var(), 2.0)

    def test_nct_inf_moments(self):
        # 对于给定的 df 和 nc 值，计算非中心 t 分布的一些统计量：m, v, s, k
        m, v, s, k = stats.nct.stats(df=0.9, nc=0.3, moments='mvsk')
        # 断言 m, v, s, k 的计算结果为 np.nan
        assert_equal([m, v, s, k], [np.nan, np.nan, np.nan, np.nan])

        m, v, s, k = stats.nct.stats(df=1.9, nc=0.3, moments='mvsk')
        # 断言 m 为有限值
        assert_(np.isfinite(m))
        # 断言 v, s, k 的计算结果为 np.nan
        assert_equal([v, s, k], [np.nan, np.nan, np.nan])

        m, v, s, k = stats.nct.stats(df=3.1, nc=0.3, moments='mvsk')
        # 断言 m, v, s 为有限值
        assert_(np.isfinite([m, v, s]).all())
        # 断言 k 的计算结果为 np.nan
        assert_equal(k, np.nan)

    def test_nct_stats_large_df_values(self):
        # 先前使用 gamma 函数在 df=345 时失去精度
        # 参见 https://github.com/scipy/scipy/issues/12919 获取详细信息
        # 计算 df=1000, nc=2 的非中心 t 分布的均值和统计量
        nct_mean_df_1000 = stats.nct.mean(1000, 2)
        nct_stats_df_1000 = stats.nct.stats(1000, 2)
        # 预期的均值和统计量，通过 mpmath 计算得出
        expected_stats_df_1000 = [2.0015015641422464, 1.0040115288163005]
        # 断言 nct_mean_df_1000 接近于 expected_stats_df_1000[0]，相对误差容差为 1e-10
        assert_allclose(nct_mean_df_1000, expected_stats_df_1000[0],
                        rtol=1e-10)
        # 断言 nct_stats_df_1000 接近于 expected_stats_df_1000，相对误差容差为 1e-10
        assert_allclose(nct_stats_df_1000, expected_stats_df_1000,
                        rtol=1e-10)
        # 计算 df=100000, nc=2 的非中心 t 分布的均值和统计量
        nct_mean = stats.nct.mean(100000, 2)
        nct_stats = stats.nct.stats(100000, 2)
        # 预期的均值和统计量，通过 mpmath 计算得出
        expected_stats = [2.0000150001562518, 1.0000400011500288]
        # 断言 nct_mean 接近于 expected_stats[0]，相对误差容差为 1e-10
        assert_allclose(nct_mean, expected_stats[0], rtol=1e-10)
        # 断言 nct_stats 接近于 expected_stats，相对误差容差为 1e-9
        assert_allclose(nct_stats, expected_stats, rtol=1e-9)
    def test_cdf_large_nc(self):
        # 定义一个单元测试函数，测试累积分布函数对于大 `nc` 值的行为
        # gh-17916 报告了在大 `nc` 值时的崩溃情况
        assert_allclose(stats.nct.cdf(2, 2, float(2**16)), 0)
class TestRecipInvGauss:

    def test_pdf_endpoint(self):
        # 计算逆高斯分布的概率密度函数在端点值处（x=0, mu=0.6）
        p = stats.recipinvgauss.pdf(0, 0.6)
        assert p == 0.0

    def test_logpdf_endpoint(self):
        # 计算逆高斯分布的对数概率密度函数在端点值处（x=0, mu=0.6）
        logp = stats.recipinvgauss.logpdf(0, 0.6)
        assert logp == -np.inf

    def test_cdf_small_x(self):
        # 用 mpmath 计算的期望值：
        #
        # import mpmath
        # mpmath.mp.dps = 100
        # def recipinvgauss_cdf_mp(x, mu):
        #     x = mpmath.mpf(x)
        #     mu = mpmath.mpf(mu)
        #     trm1 = 1/mu - x
        #     trm2 = 1/mu + x
        #     isqx = 1/mpmath.sqrt(x)
        #     return (mpmath.ncdf(-isqx*trm1)
        #             - mpmath.exp(2/mu)*mpmath.ncdf(-isqx*trm2))
        #
        # 计算逆高斯分布的累积分布函数在小值 x=0.05, mu=0.5 处
        p = stats.recipinvgauss.cdf(0.05, 0.5)
        expected = 6.590396159501331e-20
        assert_allclose(p, expected, rtol=1e-14)

    def test_sf_large_x(self):
        # 用 mpmath 计算的期望值；参见 test_cdf_small。
        # 计算逆高斯分布的生存函数在大值 x=80, mu=0.5 处
        p = stats.recipinvgauss.sf(80, 0.5)
        expected = 2.699819200556787e-18
        assert_allclose(p, expected, 5e-15)


class TestRice:
    def test_rice_zero_b(self):
        # 当 b=0 时，稻谷分布应该正常工作，参见 gh-2164
        x = [0.2, 1., 5.]
        assert_(np.isfinite(stats.rice.pdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.logpdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.cdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.logcdf(x, b=0.)).all())

        q = [0.1, 0.1, 0.5, 0.9]
        assert_(np.isfinite(stats.rice.ppf(q, b=0.)).all())

        mvsk = stats.rice.stats(0, moments='mvsk')
        assert_(np.isfinite(mvsk).all())

        # 此外，当 b\to 0 时，概率密度函数连续
        # rice.pdf(x, b\to 0) = x exp(-x^2/2) + O(b^2)
        # 参见 Abramovich & Stegun 9.6.7 & 9.6.10
        b = 1e-8
        assert_allclose(stats.rice.pdf(x, 0), stats.rice.pdf(x, b),
                        atol=b, rtol=0)

    def test_rice_rvs(self):
        rvs = stats.rice.rvs
        assert_equal(rvs(b=3.).size, 1)
        assert_equal(rvs(b=3., size=(3, 5)).shape, (3, 5))
    def test_rice_gh9836(self):
        # 测试解决 gh-9836；先前在末尾跳到 1 的问题

        # 使用 Rice 分布的累积分布函数（CDF）计算，参数为从 10 到 150，步长为 10 的数组
        cdf = stats.rice.cdf(np.arange(10, 160, 10), np.arange(10, 160, 10))
        # 以下是在 R 中生成的期望结果
        # library(VGAM)
        # options(digits=16)
        # x = seq(10, 150, 10)
        # print(price(x, sigma=1, vee=x))
        cdf_exp = [0.4800278103504522, 0.4900233218590353, 0.4933500379379548,
                   0.4950128317658719, 0.4960103776798502, 0.4966753655438764,
                   0.4971503395812474, 0.4975065620443196, 0.4977836197921638,
                   0.4980052636649550, 0.4981866072661382, 0.4983377260666599,
                   0.4984655952615694, 0.4985751970541413, 0.4986701850071265]
        # 使用 assert_allclose 函数断言计算得到的 cdf 与期望结果 cdf_exp 很接近
        assert_allclose(cdf, cdf_exp)

        # 使用 Rice 分布的百分位点函数（PPF）计算，参数为从 0.1 到 0.9，步长为 0.1 的数组
        ppf = stats.rice.ppf(probabilities, 500/4, scale=4)
        # 以下是在 R 中生成的期望结果
        # library(VGAM)
        # options(digits=16)
        # p = seq(0.1, .9, by = .1)
        # print(qrice(p, vee = 500, sigma = 4))
        ppf_exp = [494.8898762347361, 496.6495690858350, 497.9184315188069,
                   499.0026277378915, 500.0159999146250, 501.0293721352668,
                   502.1135684981884, 503.3824312270405, 505.1421247157822]
        # 使用 assert_allclose 函数断言计算得到的 ppf 与期望结果 ppf_exp 很接近
        assert_allclose(ppf, ppf_exp)

        # 使用 Rice 分布的百分位点函数（PPF）计算，参数为从 10 到 140，步长为 10 的数组
        ppf = scipy.stats.rice.ppf(0.5, np.arange(10, 150, 10))
        # 以下是在 R 中生成的期望结果
        # library(VGAM)
        # options(digits=16)
        # b <- seq(10, 140, 10)
        # print(qrice(0.5, vee = b, sigma = 1))
        ppf_exp = [10.04995862522287, 20.02499480078302, 30.01666512465732,
                   40.01249934924363, 50.00999966676032, 60.00833314046875,
                   70.00714273568241, 80.00624991862573, 90.00555549840364,
                   100.00499995833597, 110.00454542324384, 120.00416664255323,
                   130.00384613488120, 140.00357141338748]
        # 使用 assert_allclose 函数断言计算得到的 ppf 与期望结果 ppf_exp 很接近
        assert_allclose(ppf, ppf_exp)
# 定义一个测试类 TestErlang，用于测试 Erlang 分布的相关功能
class TestErlang:
    
    # 在每个测试方法运行前设置随机种子为 1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试非整数形状参数是否会触发 RuntimeWarning
    def test_erlang_runtimewarning(self):
        # 捕获 RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            
            # 使用非整数形状参数 1.3 应该触发 RuntimeWarning
            assert_raises(RuntimeWarning,
                          stats.erlang.rvs, 1.3, loc=0, scale=1, size=4)
            
            # 使用整数形状参数调用 fit 方法不应该触发 RuntimeWarning，
            # 应该返回与 gamma.fit(...) 相同的值
            data = [0.5, 1.0, 2.0, 4.0]
            result_erlang = stats.erlang.fit(data, f0=1)
            result_gamma = stats.gamma.fit(data, f0=1)
            assert_allclose(result_erlang, result_gamma, rtol=1e-3)

    # 测试 gh-10949 中的参数检查
    def test_gh_pr_10949_argcheck(self):
        assert_equal(stats.erlang.pdf(0.5, a=[1, -1]),
                     stats.gamma.pdf(0.5, a=[1, -1]))


# 定义一个测试类 TestRayleigh，用于测试 Rayleigh 分布的相关功能
class TestRayleigh:
    
    # 在每个测试方法运行前设置随机种子为 987654321
    def setup_method(self):
        np.random.seed(987654321)

    # 测试 logpdf 方法
    def test_logpdf(self):
        y = stats.rayleigh.logpdf(50)
        assert_allclose(y, -1246.0879769945718)

    # 测试 logsf 方法
    def test_logsf(self):
        y = stats.rayleigh.logsf(50)
        assert_allclose(y, -1250)

    # 参数化测试 fit 方法的不同情况
    @pytest.mark.parametrize("rvs_loc,rvs_scale", [(0.85373171, 0.86932204),
                                                   (0.20558821, 0.61621008)])
    def test_fit(self, rvs_loc, rvs_scale):
        # 生成 Rayleigh 分布的随机样本数据
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)

        # 定义 scale_mle 函数用于估计尺度参数
        def scale_mle(data, floc):
            return (np.sum((data - floc) ** 2) / (2 * len(data))) ** .5

        # 当提供 `floc` 参数时，使用解析公式找到 `scale` 参数
        scale_expect = scale_mle(data, rvs_loc)
        loc, scale = stats.rayleigh.fit(data, floc=rvs_loc)
        assert_equal(loc, rvs_loc)
        assert_equal(scale, scale_expect)

        # 当 `fscale` 固定时，使用超类的 fit 方法确定 `loc` 参数
        loc, scale = stats.rayleigh.fit(data, fscale=.6)
        assert_equal(scale, .6)

        # 当两个参数都自由时，进行一维优化，考虑 `scale` 与 `loc` 的依赖关系
        loc, scale = stats.rayleigh.fit(data)
        # 测试 `scale` 是否通过其与 `loc` 的关系定义
        assert_equal(scale, scale_mle(data, loc))
    # 测试比较超级方法中的拟合结果
    def test_fit_comparison_super_method(self, rvs_loc, rvs_scale):
        # 生成服从 Rayleigh 分布的随机数据
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)
        # 调用函数验证统计模型的对数似然值是否符合预期
        _assert_less_or_close_loglike(stats.rayleigh, data)

    # 测试拟合过程中的警告情况
    def test_fit_warnings(self):
        # 验证 Rayleigh 分布的拟合是否会产生警告
        assert_fit_warnings(stats.rayleigh)

    # 测试特定 issue（gh-17088）中的拟合情况
    def test_fit_gh17088(self):
        # 在特定情况下，Rayleigh 分布的拟合可能导致位置参数不一致。参见 gh-17088。
        # 设置随机数生成器和参数
        rng = np.random.default_rng(456)
        loc, scale, size = 50, 600, 500
        # 生成 Rayleigh 分布的随机变量
        rvs = stats.rayleigh.rvs(loc, scale, size=size, random_state=rng)
        
        # 使用默认参数进行拟合，验证位置参数是否小于随机变量的最小值
        loc_fit, _ = stats.rayleigh.fit(rvs)
        assert loc_fit < np.min(rvs)
        
        # 使用指定尺度参数进行拟合，同样验证位置参数是否小于随机变量的最小值，以及尺度参数是否与给定尺度相等
        loc_fit, scale_fit = stats.rayleigh.fit(rvs, fscale=scale)
        assert loc_fit < np.min(rvs)
        assert scale_fit == scale
class TestExponWeib:

    def test_pdf_logpdf(self):
        # Regression test for gh-3508.
        # 设置测试数据
        x = 0.1
        a = 1.0
        c = 100.0
        # 计算指数威布尔分布的概率密度函数值和对数概率密度函数值
        p = stats.exponweib.pdf(x, a, c)
        logp = stats.exponweib.logpdf(x, a, c)
        # 预期值是使用mpmath计算得到的
        # 检查计算结果是否接近预期值
        assert_allclose([p, logp],
                        [1.0000000000000054e-97, -223.35075402042244])

    def test_a_is_1(self):
        # For issue gh-3508.
        # 当 a=1 时，检查指数威布尔分布的概率密度函数和对数概率密度函数是否与威布尔分布相同
        x = np.logspace(-4, -1, 4)
        a = 1
        c = 100

        # 计算指数威布尔分布和威布尔分布的概率密度函数值，并进行比较
        p = stats.exponweib.pdf(x, a, c)
        expected = stats.weibull_min.pdf(x, c)
        assert_allclose(p, expected)

        # 计算指数威布尔分布和威布尔分布的对数概率密度函数值，并进行比较
        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.weibull_min.logpdf(x, c)
        assert_allclose(logp, expected)

    def test_a_is_1_c_is_1(self):
        # 当 a=1 且 c=1 时，分布为指数分布
        x = np.logspace(-8, 1, 10)
        a = 1
        c = 1

        # 计算指数威布尔分布和指数分布的概率密度函数值，并进行比较
        p = stats.exponweib.pdf(x, a, c)
        expected = stats.expon.pdf(x)
        assert_allclose(p, expected)

        # 计算指数威布尔分布和指数分布的对数概率密度函数值，并进行比较
        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.expon.logpdf(x)
        assert_allclose(logp, expected)

    # 参考值使用 mpmath 计算得到，例如：
    #
    #     from mpmath import mp
    #
    #     def mp_sf(x, a, c):
    #         x = mp.mpf(x)
    #         a = mp.mpf(a)
    #         c = mp.mpf(c)
    #         return -mp.powm1(-mp.expm1(-x**c), a)
    #
    #     mp.dps = 100
    #     print(float(mp_sf(1, 2.5, 0.75)))
    #
    # 输出
    #
    #     0.6823127476985246
    #
    @pytest.mark.parametrize(
        'x, a, c, ref',
        [(1, 2.5, 0.75, 0.6823127476985246),
         (50, 2.5, 0.75, 1.7056666054719663e-08),
         (125, 2.5, 0.75, 1.4534393150714602e-16),
         (250, 2.5, 0.75, 1.2391389689773512e-27),
         (250, 0.03125, 0.75, 1.548923711221689e-29),
         (3, 0.03125, 3.0,  5.873527551689983e-14),
         (2e80, 10.0, 0.02, 2.9449084156902135e-17)]
    )
    def test_sf(self, x, a, c, ref):
        # 计算生存函数（Survival function）并检查是否接近参考值
        sf = stats.exponweib.sf(x, a, c)
        assert_allclose(sf, ref, rtol=1e-14)

    # 参考值使用 mpmath 计算得到，例如：
    #
    #     from mpmath import mp
    #
    #     def mp_isf(p, a, c):
    #         p = mp.mpf(p)
    #         a = mp.mpf(a)
    #         c = mp.mpf(c)
    #         return (-mp.log(-mp.expm1(mp.log1p(-p)/a)))**(1/c)
    #
    #     mp.dps = 100
    #     print(float(mp_isf(0.25, 2.5, 0.75)))
    #
    # 输出
    #
    #     2.8946008178158924
    #
    @pytest.mark.parametrize(
        'p, a, c, ref',
        [(0.25, 2.5, 0.75, 2.8946008178158924),
         (3e-16, 2.5, 0.75, 121.77966713102938),
         (1e-12, 1, 2, 5.256521769756932),
         (2e-13, 0.03125, 3, 2.953915059484589),
         (5e-14, 10.0, 0.02, 7.57094886384687e+75)]
    )
    def test_isf(self, p, a, c, ref):
        # 计算逆生存函数（Inverse survival function）并检查是否接近参考值
        isf = stats.exponweib.isf(p, a, c)
        assert_allclose(isf, ref)
    # 定义一个测试函数，用于验证指数威布尔分布的逆累积分布函数（ISF）
    def test_isf(self, p, a, c, ref):
        # 计算指数威布尔分布的逆累积分布函数值
        isf = stats.exponweib.isf(p, a, c)
        # 断言计算得到的逆累积分布函数值与参考值 ref 相近，相对容差为 5e-14
        assert_allclose(isf, ref, rtol=5e-14)
class TestFatigueLife:

    def test_sf_tail(self):
        # 用于测试生命疲劳分布的生存函数（Survival Function）计算
        # 预期值是使用mpmath计算的：
        #     import mpmath
        #     mpmath.mp.dps = 80
        #     x = mpmath.mpf(800.0)
        #     c = mpmath.mpf(2.5)
        #     s = float(1 - mpmath.ncdf(1/c * (mpmath.sqrt(x)
        #                                      - 1/mpmath.sqrt(x))))
        #     print(s)
        # 输出:
        #     6.593376447038406e-30
        s = stats.fatiguelife.sf(800.0, 2.5)
        assert_allclose(s, 6.593376447038406e-30, rtol=1e-13)

    def test_isf_tail(self):
        # 参考test_sf_tail中的mpmath代码。
        # 测试生命疲劳分布的逆生存函数（Inverse Survival Function）计算
        p = 6.593376447038406e-30
        q = stats.fatiguelife.isf(p, 2.5)
        assert_allclose(q, 800.0, rtol=1e-13)


class TestWeibull:

    def test_logpdf(self):
        # gh-6217
        # 测试威布尔分布的对数概率密度函数（Log Probability Density Function）
        y = stats.weibull_min.logpdf(0, 1)
        assert_equal(y, 0)

    @pytest.mark.parametrize('scale', [1.0, 0.1])
    def test_delta_cdf(self, scale):
        # 使用mpmath计算的预期值：
        #
        # def weibull_min_sf(x, k, scale):
        #     x = mpmath.mpf(x)
        #     k = mpmath.mpf(k)
        #     scale =mpmath.mpf(scale)
        #     return mpmath.exp(-(x/scale)**k)
        #
        # >>> import mpmath
        # >>> mpmath.mp.dps = 60
        # >>> sf1 = weibull_min_sf(7.5, 3, 1)
        # >>> sf2 = weibull_min_sf(8.0, 3, 1)
        # >>> float(sf1 - sf2)
        # 6.053624060118734e-184
        #
        # 测试威布尔分布的累积分布函数差值（Cumulative Distribution Function Difference）
        delta = stats.weibull_min._delta_cdf(scale*7.5, scale*8, 3,
                                             scale=scale)
        assert_allclose(delta, 6.053624060118734e-184)

    def test_fit_min(self):
        rng = np.random.default_rng(5985959307161735394)

        c, loc, scale = 2, 3.5, 0.5  # 任意选择的有效参数
        dist = stats.weibull_min(c, loc, scale)
        rvs = dist.rvs(size=100, random_state=rng)

        # 测试最大似然估计是否仍然尊重猜测和固定参数
        c2, loc2, scale2 = stats.weibull_min.fit(rvs, 1.5, floc=3)
        c3, loc3, scale3 = stats.weibull_min.fit(rvs, 1.6, floc=3)
        assert loc2 == loc3 == 3  # 固定参数被尊重
        assert c2 != c3  # 不同的猜测导致（稍微）不同的结果
        # 拟合质量在其他地方进行测试

        # 测试矩法估计是否尊重固定参数，接受（但忽略）猜测
        c4, loc4, scale4 = stats.weibull_min.fit(rvs, 3, fscale=3, method='mm')
        assert scale4 == 3
        # 因为尺度被固定，所以只匹配均值和偏度
        dist4 = stats.weibull_min(c4, loc4, scale4)
        res = dist4.stats(moments='ms')
        ref = np.mean(rvs), stats.skew(rvs)
        assert_allclose(res, ref)

    # 参考值是通过mpmath计算的
    # from mpmath import mp
    # def weibull_sf_mpmath(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return float(mp.exp(-x**c))
    # 使用 pytest 的参数化装饰器标记此测试方法，用于多组参数的测试
    @pytest.mark.parametrize('x, c, ref', [(50, 1, 1.9287498479639178e-22),
                                           (1000, 0.8,
                                            8.131269637872743e-110)])
    # 定义测试方法，测试 Weibull 分布的生存函数（sf）和反生存函数（isf）
    def test_sf_isf(self, x, c, ref):
        # 断言 Weibull 分布的生存函数（sf）计算结果与参考值相近
        assert_allclose(stats.weibull_min.sf(x, c), ref, rtol=5e-14)
        # 断言 Weibull 分布的反生存函数（isf）计算结果与 x 相近
        assert_allclose(stats.weibull_min.isf(ref, c), x, rtol=5e-14)
class TestDweibull:
    def test_entropy(self):
        # 测试 dweibull 熵是否遵循 weibull_min 的熵。
        # （通用测试检查 dweibull 熵是否与其概率密度函数一致。
        #  关于准确性，dweibull 熵应该与 weibull_min 熵一样准确。
        #  对于准确性的检查只需应用于基本分布 - weibull_min。）
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(8486259129157041777)
        # 从正态分布中获取标准差为100的10个随机数，并乘以10的幂级数
        c = 10**rng.normal(scale=100, size=10)
        # 计算 dweibull 熵
        res = stats.dweibull.entropy(c)
        # 计算参考的 weibull_min 熵，减去 ln(0.5)
        ref = stats.weibull_min.entropy(c) - np.log(0.5)
        # 检查结果是否在指定相对误差内接近
        assert_allclose(res, ref, rtol=1e-15)

    def test_sf(self):
        # 测试对于正值，dweibull 生存函数是否为 weibull_min 生存函数的一半
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(8486259129157041777)
        # 从正态分布中获取标准差为1的10个随机数，并乘以10的幂级数
        c = 10**rng.normal(scale=1, size=10)
        # 获取随机数并将其乘以10
        x = 10 * rng.uniform()
        # 计算 dweibull 生存函数
        res = stats.dweibull.sf(x, c)
        # 计算参考的 weibull_min 生存函数的一半
        ref = 0.5 * stats.weibull_min.sf(x, c)
        # 检查结果是否在指定相对误差内接近
        assert_allclose(res, ref, rtol=1e-15)


class TestTruncWeibull:

    def test_pdf_bounds(self):
        # 测试边界条件
        # 计算给定参数下的概率密度函数值
        y = stats.truncweibull_min.pdf([0.1, 2.0], 2.0, 0.11, 1.99)
        # 检查计算结果是否与期望值相等
        assert_equal(y, [0.0, 0.0])

    def test_logpdf(self):
        # 计算对数概率密度函数值
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, np.inf)
        # 检查计算结果是否等于期望值
        assert_equal(y, 0.0)

        # 手动计算
        # 计算对数概率密度函数值
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, 4.0)
        # 检查计算结果是否在指定误差内接近期望值
        assert_allclose(y, 0.14541345786885884)

    def test_ppf_bounds(self):
        # 测试边界条件
        # 计算给定参数下的百分点函数值
        y = stats.truncweibull_min.ppf([0.0, 1.0], 2.0, 0.1, 2.0)
        # 检查计算结果是否与期望值相等
        assert_equal(y, [0.1, 2.0])

    def test_cdf_to_ppf(self):
        # 将累积分布函数值转换为百分点函数值
        q = [0., 0.1, .25, 0.50, 0.75, 0.90, 1.]
        # 计算给定参数下的百分点函数值
        x = stats.truncweibull_min.ppf(q, 2., 0., 3.)
        # 计算给定参数下的累积分布函数值
        q_out = stats.truncweibull_min.cdf(x, 2., 0., 3.)
        # 检查计算结果是否在指定误差内接近期望值
        assert_allclose(q, q_out)

    def test_sf_to_isf(self):
        # 将生存函数值转换为逆生存函数值
        q = [0., 0.1, .25, 0.50, 0.75, 0.90, 1.]
        # 计算给定参数下的逆生存函数值
        x = stats.truncweibull_min.isf(q, 2., 0., 3.)
        # 计算给定参数下的生存函数值
        q_out = stats.truncweibull_min.sf(x, 2., 0., 3.)
        # 检查计算结果是否在指定误差内接近期望值
        assert_allclose(q, q_out)
    # 定义一个测试函数，用于验证修剪威布尔分布的统计函数的准确性
    def test_munp(self):
        # 设置修剪参数 c
        c = 2.
        # 设置修剪威布尔分布的参数 a
        a = 1.
        # 设置修剪威布尔分布的参数 b
        b = 3.

        # 定义一个内部函数 xnpdf，用于计算修剪威布尔分布的概率密度函数值
        def xnpdf(x, n):
            return x**n * stats.truncweibull_min.pdf(x, c, a, b)

        # 计算修剪威布尔分布的零阶矩
        m0 = stats.truncweibull_min.moment(0, c, a, b)
        # 断言零阶矩的计算结果为 1.0
        assert_equal(m0, 1.)

        # 计算修剪威布尔分布的一阶矩
        m1 = stats.truncweibull_min.moment(1, c, a, b)
        # 使用数值积分计算期望的一阶矩
        m1_expected, _ = quad(lambda x: xnpdf(x, 1), a, b)
        # 断言修剪威布尔分布的一阶矩与数值积分计算的期望一阶矩相近
        assert_allclose(m1, m1_expected)

        # 计算修剪威布尔分布的二阶矩
        m2 = stats.truncweibull_min.moment(2, c, a, b)
        # 使用数值积分计算期望的二阶矩
        m2_expected, _ = quad(lambda x: xnpdf(x, 2), a, b)
        # 断言修剪威布尔分布的二阶矩与数值积分计算的期望二阶矩相近
        assert_allclose(m2, m2_expected)

        # 计算修剪威布尔分布的三阶矩
        m3 = stats.truncweibull_min.moment(3, c, a, b)
        # 使用数值积分计算期望的三阶矩
        m3_expected, _ = quad(lambda x: xnpdf(x, 3), a, b)
        # 断言修剪威布尔分布的三阶矩与数值积分计算的期望三阶矩相近
        assert_allclose(m3, m3_expected)

        # 计算修剪威布尔分布的四阶矩
        m4 = stats.truncweibull_min.moment(4, c, a, b)
        # 使用数值积分计算期望的四阶矩
        m4_expected, _ = quad(lambda x: xnpdf(x, 4), a, b)
        # 断言修剪威布尔分布的四阶矩与数值积分计算的期望四阶矩相近
        assert_allclose(m4, m4_expected)

    # 定义一个测试函数，用于验证修剪威布尔分布的统计函数返回的参考值的准确性
    def test_reference_values(self):
        # 设置修剪威布尔分布的参数 a
        a = 1.
        # 设置修剪威布尔分布的参数 b
        b = 3.
        # 设置修剪参数 c
        c = 2.
        # 计算修剪威布尔分布的中位数
        x_med = np.sqrt(1 - np.log(0.5 + np.exp(-(8. + np.log(2.)))))

        # 计算修剪威布尔分布的累积分布函数值
        cdf = stats.truncweibull_min.cdf(x_med, c, a, b)
        # 断言修剪威布尔分布的累积分布函数值为 0.5
        assert_allclose(cdf, 0.5)

        # 计算修剪威布尔分布的对数累积分布函数值
        lc = stats.truncweibull_min.logcdf(x_med, c, a, b)
        # 断言修剪威布尔分布的对数累积分布函数值为 -log(2)
        assert_allclose(lc, -np.log(2.))

        # 计算修剪威布尔分布的累积分布函数的反函数值
        ppf = stats.truncweibull_min.ppf(0.5, c, a, b)
        # 断言修剪威布尔分布的累积分布函数的反函数值为中位数
        assert_allclose(ppf, x_med)

        # 计算修剪威布尔分布的生存函数值
        sf = stats.truncweibull_min.sf(x_med, c, a, b)
        # 断言修剪威布尔分布的生存函数值为 0.5
        assert_allclose(sf, 0.5)

        # 计算修剪威布尔分布的对数生存函数值
        ls = stats.truncweibull_min.logsf(x_med, c, a, b)
        # 断言修剪威布尔分布的对数生存函数值为 -log(2)
        assert_allclose(ls, -np.log(2.))

        # 计算修剪威布尔分布的生存函数的反函数值
        isf = stats.truncweibull_min.isf(0.5, c, a, b)
        # 断言修剪威布尔分布的生存函数的反函数值为中位数
        assert_allclose(isf, x_med)
    def test_compare_weibull_min(self):
        # 验证 truncweibull_min 分布是否与原始 weibull_min 给出相同结果

        # 定义参数
        x = 1.5
        c = 2.0
        a = 0.0
        b = np.inf
        scale = 3.0

        # 计算 weibull_min 分布的概率密度函数值
        p = stats.weibull_min.pdf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的概率密度函数值
        p_trunc = stats.truncweibull_min.pdf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(p, p_trunc)

        # 计算 weibull_min 分布的对数概率密度函数值
        lp = stats.weibull_min.logpdf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的对数概率密度函数值
        lp_trunc = stats.truncweibull_min.logpdf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(lp, lp_trunc)

        # 计算 weibull_min 分布的累积分布函数值
        cdf = stats.weibull_min.cdf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的累积分布函数值
        cdf_trunc = stats.truncweibull_min.cdf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(cdf, cdf_trunc)

        # 计算 weibull_min 分布的对数累积分布函数值
        lc = stats.weibull_min.logcdf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的对数累积分布函数值
        lc_trunc = stats.truncweibull_min.logcdf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(lc, lc_trunc)

        # 计算 weibull_min 分布的生存函数值
        s = stats.weibull_min.sf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的生存函数值
        s_trunc = stats.truncweibull_min.sf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(s, s_trunc)

        # 计算 weibull_min 分布的对数生存函数值
        ls = stats.weibull_min.logsf(x, c, scale=scale)
        # 计算 truncweibull_min 分布的对数生存函数值
        ls_trunc = stats.truncweibull_min.logsf(x, c, a, b, scale=scale)
        # 检查两者是否接近
        assert_allclose(ls, ls_trunc)

        # 也测试当 x 值较大时，通过累积分布函数计算生存函数的情况，
        # 这种情况下，生存函数值接近 0
        s = stats.truncweibull_min.sf(30, 2, a, b, scale=3)
        assert_allclose(s, np.exp(-100))

        # 对数生存函数值也应该接近 -100
        ls = stats.truncweibull_min.logsf(30, 2, a, b, scale=3)
        assert_allclose(ls, -100)

    def test_compare_weibull_min2(self):
        # 验证 truncweibull_min 分布的概率密度函数和累积分布函数结果
        # 是否与截断 weibull_min 分布的计算结果相同

        # 定义参数
        c, a, b = 2.5, 0.25, 1.25
        x = np.linspace(a, b, 100)

        # 计算 truncweibull_min 分布的概率密度函数值
        pdf1 = stats.truncweibull_min.pdf(x, c, a, b)
        # 计算 truncweibull_min 分布的累积分布函数值
        cdf1 = stats.truncweibull_min.cdf(x, c, a, b)

        # 计算未截断的 weibull_min 分布在相同区间的概率密度函数值
        norm = stats.weibull_min.cdf(b, c) - stats.weibull_min.cdf(a, c)
        pdf2 = stats.weibull_min.pdf(x, c) / norm
        # 计算未截断的 weibull_min 分布在相同区间的累积分布函数值
        cdf2 = (stats.weibull_min.cdf(x, c) - stats.weibull_min.cdf(a, c))/norm

        # 检查两者是否接近
        np.testing.assert_allclose(pdf1, pdf2)
        np.testing.assert_allclose(cdf1, cdf2)
class TestRdist:
    def test_rdist_cdf_gh1285(self):
        # 检查在 rdist._cdf 中针对问题 gh-1285 的解决方法。
        distfn = stats.rdist  # 获取 rdist 函数对象
        values = [0.001, 0.5, 0.999]  # 定义测试值列表
        assert_almost_equal(distfn.cdf(distfn.ppf(values, 541.0), 541.0),
                            values, decimal=5)  # 断言两者之间的近似相等性

    def test_rdist_beta(self):
        # rdist 是 stats.beta 的特殊情况
        x = np.linspace(-0.99, 0.99, 10)  # 在区间[-0.99, 0.99]上生成等间距的10个点作为 x
        c = 2.7  # 设定参数 c
        assert_almost_equal(0.5*stats.beta(c/2, c/2).pdf((x + 1)/2),
                            stats.rdist(c).pdf(x))  # 断言 rdist(c).pdf(x) 与 0.5*stats.beta(c/2, c/2).pdf((x + 1)/2) 的近似相等性

    # reference values were computed via mpmath
    # from mpmath import mp
    # mp.dps = 200
    # def rdist_sf_mpmath(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return float(mp.betainc(c/2, c/2, (x+1)/2, mp.one, regularized=True))
    @pytest.mark.parametrize(
        "x, c, ref",
        [
            (0.0001, 541, 0.49907251345565845),
            (0.1, 241, 0.06000788166249205),
            (0.5, 441, 1.0655898106047832e-29),
            (0.8, 341, 6.025478373732215e-78),
        ]
    )
    def test_rdist_sf(self, x, c, ref):
        assert_allclose(stats.rdist.sf(x, c), ref, rtol=5e-14)  # 断言 stats.rdist.sf(x, c) 与 ref 的绝对误差在允许范围内


class TestTrapezoid:
    def test_reduces_to_triang(self):
        modes = [0, 0.3, 0.5, 1]  # 定义 modes 列表
        for mode in modes:
            x = [0, mode, 1]  # 定义 x 列表
            assert_almost_equal(stats.trapezoid.pdf(x, mode, mode),
                                stats.triang.pdf(x, mode))  # 断言 trapezoid.pdf(x, mode, mode) 与 triang.pdf(x, mode) 的近似相等性
            assert_almost_equal(stats.trapezoid.cdf(x, mode, mode),
                                stats.triang.cdf(x, mode))  # 断言 trapezoid.cdf(x, mode, mode) 与 triang.cdf(x, mode) 的近似相等性

    def test_reduces_to_uniform(self):
        x = np.linspace(0, 1, 10)  # 在区间[0, 1]上生成等间距的10个点作为 x
        assert_almost_equal(stats.trapezoid.pdf(x, 0, 1), stats.uniform.pdf(x))  # 断言 trapezoid.pdf(x, 0, 1) 与 uniform.pdf(x) 的近似相等性
        assert_almost_equal(stats.trapezoid.cdf(x, 0, 1), stats.uniform.cdf(x))  # 断言 trapezoid.cdf(x, 0, 1) 与 uniform.cdf(x) 的近似相等性

    def test_cases(self):
        # edge cases
        assert_almost_equal(stats.trapezoid.pdf(0, 0, 0), 2)  # 断言 trapezoid.pdf(0, 0, 0) 的近似相等性为 2
        assert_almost_equal(stats.trapezoid.pdf(1, 1, 1), 2)  # 断言 trapezoid.pdf(1, 1, 1) 的近似相等性为 2
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0, 0.8),
                            1.11111111111111111)  # 断言 trapezoid.pdf(0.5, 0, 0.8) 的近似相等性
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0.2, 1.0),
                            1.11111111111111111)  # 断言 trapezoid.pdf(0.5, 0.2, 1.0) 的近似相等性

        # straightforward case
        assert_almost_equal(stats.trapezoid.pdf(0.1, 0.2, 0.8), 0.625)  # 断言 trapezoid.pdf(0.1, 0.2, 0.8) 的近似相等性为 0.625
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0.2, 0.8), 1.25)  # 断言 trapezoid.pdf(0.5, 0.2, 0.8) 的近似相等性为 1.25
        assert_almost_equal(stats.trapezoid.pdf(0.9, 0.2, 0.8), 0.625)  # 断言 trapezoid.pdf(0.9, 0.2, 0.8) 的近似相等性为 0.625

        assert_almost_equal(stats.trapezoid.cdf(0.1, 0.2, 0.8), 0.03125)  # 断言 trapezoid.cdf(0.1, 0.2, 0.8) 的近似相等性为 0.03125
        assert_almost_equal(stats.trapezoid.cdf(0.2, 0.2, 0.8), 0.125)  # 断言 trapezoid.cdf(0.2, 0.2, 0.8) 的近似相等性为 0.125
        assert_almost_equal(stats.trapezoid.cdf(0.5, 0.2, 0.8), 0.5)  # 断言 trapezoid.cdf(0.5, 0.2, 0.8) 的近似相等性为 0.5
        assert_almost_equal(stats.trapezoid.cdf(0.9, 0.2, 0.8), 0.96875)  # 断言 trapezoid.cdf(0.9, 0.2, 0.8) 的近似相等性为 0.96875
        assert_almost_equal(stats.trapezoid.cdf(1.0, 0.2, 0.8), 1.0)  # 断言 trapezoid.cdf(1.0, 0.2, 0.8) 的近似相等性为 1.0
    def test_moments_and_entropy(self):
        # issue #11795: improve precision of trapezoid stats
        # Apply formulas from Wikipedia for the following parameters:
        a, b, c, d = -3, -1, 2, 3  # 定义参数 a, b, c, d，并给出它们的值
        p1, p2, loc, scale = (b-a) / (d-a), (c-a) / (d-a), a, d-a  # 计算梯形分布的概率参数和位置参数
        h = 2 / (d+c-b-a)  # 计算梯形分布的尺度参数

        def moment(n):
            return (h * ((d**(n+2) - c**(n+2)) / (d-c)
                         - (b**(n+2) - a**(n+2)) / (b-a)) /
                    (n+1) / (n+2))  # 计算梯形分布的矩

        mean = moment(1)  # 计算均值
        var = moment(2) - mean**2  # 计算方差
        entropy = 0.5 * (d-c+b-a) / (d+c-b-a) + np.log(0.5 * (d+c-b-a))  # 计算熵
        assert_almost_equal(stats.trapezoid.mean(p1, p2, loc, scale),
                            mean, decimal=13)  # 断言均值的准确性
        assert_almost_equal(stats.trapezoid.var(p1, p2, loc, scale),
                            var, decimal=13)  # 断言方差的准确性
        assert_almost_equal(stats.trapezoid.entropy(p1, p2, loc, scale),
                            entropy, decimal=13)  # 断言熵的准确性

        # Check boundary cases where scipy d=0 or d=1.
        assert_almost_equal(stats.trapezoid.mean(0, 0, -3, 6), -1, decimal=13)  # 检查边界情况下的均值
        assert_almost_equal(stats.trapezoid.mean(0, 1, -3, 6), 0, decimal=13)   # 检查边界情况下的均值
        assert_almost_equal(stats.trapezoid.var(0, 1, -3, 6), 3, decimal=13)    # 检查边界情况下的方差

    def test_trapezoid_vect(self):
        # test that array-valued shapes and arguments are handled
        c = np.array([0.1, 0.2, 0.3])
        d = np.array([0.5, 0.6])[:, None]
        x = np.array([0.15, 0.25, 0.9])
        v = stats.trapezoid.pdf(x, c, d)  # 计算梯形分布的概率密度函数值

        cc, dd, xx = np.broadcast_arrays(c, d, x)

        res = np.empty(xx.size, dtype=xx.dtype)
        ind = np.arange(xx.size)
        for i, x1, c1, d1 in zip(ind, xx.ravel(), cc.ravel(), dd.ravel()):
            res[i] = stats.trapezoid.pdf(x1, c1, d1)  # 逐个计算梯形分布的概率密度函数值并存储结果

        assert_allclose(v, res.reshape(v.shape), atol=1e-15)  # 检查计算结果的一致性，使用较小的绝对容差

        # Check that the stats() method supports vector arguments.
        v = np.asarray(stats.trapezoid.stats(c, d, moments="mvsk"))  # 计算梯形分布的统计量
        cc, dd = np.broadcast_arrays(c, d)
        res = np.empty((cc.size, 4))  # 每个数值返回4个统计量
        ind = np.arange(cc.size)
        for i, c1, d1 in zip(ind, cc.ravel(), dd.ravel()):
            res[i] = stats.trapezoid.stats(c1, d1, moments="mvsk")  # 逐个计算梯形分布的统计量并存储结果

        assert_allclose(v, res.T.reshape(v.shape), atol=1e-15)  # 检查计算结果的一致性，使用较小的绝对容差

    def test_trapz(self):
        # Basic test for alias
        x = np.linspace(0, 1, 10)
        with pytest.deprecated_call(match="`trapz.pdf` is deprecated"):
            result = stats.trapz.pdf(x, 0, 1)  # 调用 trapz.pdf 方法计算概率密度函数值
        assert_almost_equal(result, stats.uniform.pdf(x))  # 断言计算结果与均匀分布的概率密度函数值的一致性

    @pytest.mark.parametrize('method', ['pdf', 'logpdf', 'cdf', 'logcdf',
                                        'sf', 'logsf', 'ppf', 'isf'])
    # 定义一个测试方法，用于测试 trapezoid 函数库中的某个方法是否已废弃
    def test_trapz_deprecation(self, method):
        # 定义两个常量 c 和 d，作为梯形积分的上下限
        c, d = 0.2, 0.8
        # 使用 getattr 函数获取 stats.trapezoid 中指定方法的预期结果
        expected = getattr(stats.trapezoid, method)(1, c, d)
        # 使用 pytest.deprecated_call 检查是否会发出关于 trapz 模块中方法已废弃的警告信息
        with pytest.deprecated_call(
            match=f"`trapz.{method}` is deprecated",
        ):
            # 使用 getattr 函数调用 stats.trapz 中指定方法，获取其结果
            result = getattr(stats.trapz, method)(1, c, d)
        # 断言实际结果与预期结果相等
        assert result == expected
class TestTriang:
    # 定义测试三角分布的测试类
    def test_edge_cases(self):
        # 测试极端情况

        # 进入一个 NumPy 错误状态上下文，确保错误会被引发
        with np.errstate(all='raise'):
            # 断言三角分布在给定参数下的概率密度函数（PDF）值
            assert_equal(stats.triang.pdf(0, 0), 2.)
            assert_equal(stats.triang.pdf(0.5, 0), 1.)
            assert_equal(stats.triang.pdf(1, 0), 0.)

            assert_equal(stats.triang.pdf(0, 1), 0)
            assert_equal(stats.triang.pdf(0.5, 1), 1.)
            assert_equal(stats.triang.pdf(1, 1), 2)

            # 断言三角分布在给定参数下的累积分布函数（CDF）值
            assert_equal(stats.triang.cdf(0., 0.), 0.)
            assert_equal(stats.triang.cdf(0.5, 0.), 0.75)
            assert_equal(stats.triang.cdf(1.0, 0.), 1.0)

            assert_equal(stats.triang.cdf(0., 1.), 0.)
            assert_equal(stats.triang.cdf(0.5, 1.), 0.25)
            assert_equal(stats.triang.cdf(1., 1.), 1)


class TestMaxwell:
    # 定义测试 Maxwell 分布的测试类

    # 参考值是用 wolfram alpha 计算得到的
    # erfc(x/sqrt(2)) + sqrt(2/pi) * x * e^(-x^2/2)

    @pytest.mark.parametrize("x, ref",
                             [(20, 2.2138865931011177e-86),
                              (0.01, 0.999999734046458435)])
    def test_sf(self, x, ref):
        # 测试 Maxwell 分布的生存函数（Survival function）
        assert_allclose(stats.maxwell.sf(x), ref, rtol=1e-14)

    # 参考值是用 wolfram alpha 计算得到的
    # sqrt(2) * sqrt(Q^(-1)(3/2, q))

    @pytest.mark.parametrize("q, ref",
                             [(0.001, 4.033142223656157022),
                              (0.9999847412109375, 0.0385743284050381),
                              (2**-55, 8.95564974719481)])
    def test_isf(self, q, ref):
        # 测试 Maxwell 分布的逆生存函数（Inverse survival function）
        assert_allclose(stats.maxwell.isf(q), ref, rtol=1e-15)


class TestMielke:
    # 定义测试 Mielke 分布的测试类
    def test_moments(self):
        # 测试矩（moments）

        k, s = 4.642, 0.597
        # 当 n < s 时，n-th 矩存在
        assert_equal(stats.mielke(k, s).moment(1), np.inf)
        assert_equal(stats.mielke(k, 1.0).moment(1), np.inf)
        assert_(np.isfinite(stats.mielke(k, 1.01).moment(1)))

    def test_burr_equivalence(self):
        # 测试与 Burr 分布等价性

        x = np.linspace(0.01, 100, 50)
        k, s = 2.45, 5.32
        # 检查 Burr 分布的概率密度函数（PDF）与 Mielke 分布的 PDF 的近似程度
        assert_allclose(stats.burr.pdf(x, s, k/s), stats.mielke.pdf(x, k, s))


class TestBurr:
    # 定义测试 Burr 分布的测试类
    def test_endpoints_7491(self):
        # 测试端点 7491

        # gh-7491
        # 计算在左端点 dst.a 处的概率密度函数（PDF）
        data = [
            [stats.fisk, (1,), 1],
            [stats.burr, (0.5, 2), 1],
            [stats.burr, (1, 1), 1],
            [stats.burr, (2, 0.5), 1],
            [stats.burr12, (1, 0.5), 0.5],
            [stats.burr12, (1, 1), 1.0],
            [stats.burr12, (1, 2), 2.0]]

        # 断言计算得到的 PDF 与预期值的近似程度
        ans = [_f.pdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [_correct_ for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)

        # 断言计算得到的对数 PDF 与预期值的近似程度
        ans = [_f.logpdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [np.log(_correct_) for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)
    def test_burr_stats_9544(self):
        # gh-9544.  Test from gh-9978
        # 定义参数 c 和 d
        c, d = 5.0, 3
        # 计算 Burr 分布的均值和方差
        mean, variance = stats.burr(c, d).stats()
        # 期望的均值和方差
        mean_hc, variance_hc = 1.4110263183925857, 0.22879948026191643
        # 断言计算得到的均值和方差与期望值相近
        assert_allclose(mean, mean_hc)
        assert_allclose(variance, variance_hc)

    def test_burr_nan_mean_var_9544(self):
        # gh-9544.  Test from gh-9978
        # 定义参数 c 和 d，以及测试空值情况下的均值和方差
        c, d = 0.5, 3
        mean, variance = stats.burr(c, d).stats()
        # 断言均值和方差为 NaN
        assert_(np.isnan(mean))
        assert_(np.isnan(variance))
        
        # 更改参数值，测试有限均值和 NaN 方差的情况
        c, d = 1.5, 3
        mean, variance = stats.burr(c, d).stats()
        assert_(np.isfinite(mean))
        assert_(np.isnan(variance))

        # 使用 stats.burr._munp 方法测试不同参数下的期望值 e1, e2, e3, e4
        c, d = 0.5, 3
        e1, e2, e3, e4 = stats.burr._munp(np.array([1, 2, 3, 4]), c, d)
        assert_(np.isnan(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        
        # 更改参数值，继续测试有限 e1 和 NaN e2, e3, e4 的情况
        c, d = 1.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        
        # 继续更改参数值，测试有限 e1, e2 和 NaN e3, e4 的情况
        c, d = 2.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        
        # 继续更改参数值，测试有限 e1, e2, e3 和 NaN e4 的情况
        c, d = 3.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isnan(e4))
        
        # 最后更改参数值，测试所有 e1, e2, e3, e4 均为有限值的情况
        c, d = 4.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isfinite(e4))

    def test_burr_isf(self):
        # 计算 Burr 分布的逆累积分布函数，并与参考值比较
        # 参考值通过参考分布计算得到
        c, d = 5.0, 3.0
        # 定义分位数列表
        q = [0.1, 1e-10, 1e-20, 1e-40]
        # 参考值列表
        ref = [1.9469686558286508, 124.57309395989076, 12457.309396155173,
               124573093.96155174]
        # 断言计算得到的逆累积分布函数值与参考值在指定相对容差下相等
        assert_allclose(stats.burr.isf(q, c, d), ref, rtol=1e-14)
class TestBurr12:

    @pytest.mark.parametrize('scale, expected',
                             [(1.0, 2.3283064359965952e-170),
                              (3.5, 5.987114417447875e-153)])
    def test_delta_cdf(self, scale, expected):
        # 使用pytest的参数化功能，为测试函数提供多组输入和期望输出
        # 这里测试了函数_stats.burr12._delta_cdf的返回值是否与期望值接近
        delta = stats.burr12._delta_cdf(2e5, 4e5, 4, 8, scale=scale)
        assert_allclose(delta, expected, rtol=1e-13)

    def test_moments_edge(self):
        # 问题gh-18838报告了burr12的矩可能无效，参见上文。
        # 在一个边缘案例中检查这是否得到解决，其中c*d == n，
        # 并将结果与Mathematica生成的结果进行比较，例如`SinghMaddalaDistribution[2, 2, 1]`在Wolfram Alpha上的输出。
        c, d = 2, 2
        mean = np.pi/4
        var = 1 - np.pi**2/16
        skew = np.pi**3/(32*var**1.5)
        kurtosis = np.nan
        ref = [mean, var, skew, kurtosis]
        # 使用assert_allclose函数检查stats.burr12(c, d).stats('mvsk')的结果是否与参考值ref接近
        res = stats.burr12(c, d).stats('mvsk')
        assert_allclose(res, ref, rtol=1e-14)

    # 参考值是使用mpmath在mp.dps = 80的设置下计算的，然后转换为float类型。
    @pytest.mark.parametrize(
        'p, c, d, ref',
        [(1e-12, 20, 0.5, 15.848931924611135),
         (1e-19, 20, 0.5, 79.43282347242815),
         (1e-12, 0.25, 35, 2.0888618213462466),
         (1e-80, 0.25, 35, 1360930951.7972188)]
    )
    def test_isf_near_zero(self, p, c, d, ref):
        # 调用stats.burr12.isf计算逆生存函数值，并使用assert_allclose检查是否接近参考值ref
        x = stats.burr12.isf(p, c, d)
        assert_allclose(x, ref, rtol=1e-14)


class TestStudentizedRange:
    # 对于alpha = .05, .01和.001，以及每个v值为[1, 3, 10, 20, 120, inf]，
    # 从每个表中选择一个Q值，其中k为[2, 8, 14, 20]。
    
    # 这些数组是以`k`作为列，`v`作为行编写的。
    # Q值取自表3：
    # https://www.jstor.org/stable/2237810
    q05 = [17.97, 45.40, 54.33, 59.56,
           4.501, 8.853, 10.35, 11.24,
           3.151, 5.305, 6.028, 6.467,
           2.950, 4.768, 5.357, 5.714,
           2.800, 4.363, 4.842, 5.126,
           2.772, 4.286, 4.743, 5.012]
    q01 = [90.03, 227.2, 271.8, 298.0,
           8.261, 15.64, 18.22, 19.77,
           4.482, 6.875, 7.712, 8.226,
           4.024, 5.839, 6.450, 6.823,
           3.702, 5.118, 5.562, 5.827,
           3.643, 4.987, 5.400, 5.645]
    q001 = [900.3, 2272, 2718, 2980,
            18.28, 34.12, 39.69, 43.05,
            6.487, 9.352, 10.39, 11.03,
            5.444, 7.313, 7.966, 8.370,
            4.772, 6.039, 6.448, 6.695,
            4.654, 5.823, 6.191, 6.411]
    qs = np.concatenate((q05, q01, q001))  # 将 q05, q01, q001 数组连接成一个新的数组 qs
    ps = [.95, .99, .999]  # 概率值列表
    vs = [1, 3, 10, 20, 120, np.inf]  # 自由度或样本大小的可能值列表，包括无穷大
    ks = [2, 8, 14, 20]  # 组数或样本数的可能值列表

    data = list(zip(product(ps, vs, ks), qs))  # 创建包含 (p, v, k) 和 q 值的元组列表 data

    # A small selection of large-v cases generated with R's `ptukey`
    # Each case is in the format (q, k, v, r_result)
    r_data = [
        (0.1, 3, 9001, 0.002752818526842),   # R 结果数据的样本列表
        (1, 10, 1000, 0.000526142388912),
        (1, 3, np.inf, 0.240712641229283),
        (4, 3, np.inf, 0.987012338626815),
        (1, 10, np.inf, 0.000519869467083),
    ]

    @pytest.mark.slow  # 标记此测试函数为慢速测试
    def test_cdf_against_tables(self):
        for pvk, q in self.data:  # 遍历 self.data 中的每个元素，其中 pvk 是 (p, v, k) 元组，q 是 q 值
            p_expected, v, k = pvk  # 解包 pvk 元组为 p_expected, v, k
            res_p = stats.studentized_range.cdf(q, k, v)  # 计算给定 q, k, v 的累积分布函数值
            assert_allclose(res_p, p_expected, rtol=1e-4)  # 断言计算出的值与期望值 p_expected 接近

    @pytest.mark.xslow  # 标记此测试函数为非常慢速测试
    def test_ppf_against_tables(self):
        for pvk, q_expected in self.data:  # 遍历 self.data 中的每个元素，其中 pvk 是 (p, v, k) 元组，q_expected 是期望的 q 值
            p, v, k = pvk  # 解包 pvk 元组为 p, v, k
            res_q = stats.studentized_range.ppf(p, k, v)  # 计算给定 p, k, v 的百分点函数值
            assert_allclose(res_q, q_expected, rtol=5e-4)  # 断言计算出的值与期望的 q 值 q_expected 接近

    path_prefix = os.path.dirname(__file__)  # 获取当前文件的目录路径
    relative_path = "data/studentized_range_mpmath_ref.json"  # 相对路径到参考数据文件
    with open(os.path.join(path_prefix, relative_path)) as file:  # 打开参考数据文件
        pregenerated_data = json.load(file)  # 加载 JSON 文件中的数据到 pregenerated_data 变量

    @pytest.mark.parametrize("case_result", pregenerated_data["cdf_data"])  # 使用参数化测试标记，遍历 pregenerated_data 中的 cdf_data
    def test_cdf_against_mp(self, case_result):
        src_case = case_result["src_case"]  # 获取测试案例的源数据
        mp_result = case_result["mp_result"]  # 获取参考实现的结果
        qkv = src_case["q"], src_case["k"], src_case["v"]  # 解包测试案例的 q, k, v 值
        res = stats.studentized_range.cdf(*qkv)  # 计算给定 q, k, v 的累积分布函数值

        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],  # 指定绝对误差容忍度
                        rtol=src_case["expected_rtol"])  # 指定相对误差容忍度

    @pytest.mark.parametrize("case_result", pregenerated_data["pdf_data"])  # 使用参数化测试标记，遍历 pregenerated_data 中的 pdf_data
    def test_pdf_against_mp(self, case_result):
        src_case = case_result["src_case"]  # 获取测试案例的源数据
        mp_result = case_result["mp_result"]  # 获取参考实现的结果
        qkv = src_case["q"], src_case["k"], src_case["v"]  # 解包测试案例的 q, k, v 值
        res = stats.studentized_range.pdf(*qkv)  # 计算给定 q, k, v 的概率密度函数值

        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],  # 指定绝对误差容忍度
                        rtol=src_case["expected_rtol"])  # 指定相对误差容忍度

    @pytest.mark.xslow  # 标记此测试函数为非常慢速测试
    @pytest.mark.xfail_on_32bit("intermittent RuntimeWarning: invalid value.")  # 在 32 位系统上，标记预期的失败和警告信息
    @pytest.mark.parametrize("case_result", pregenerated_data["moment_data"])  # 使用参数化测试标记，遍历 pregenerated_data 中的 moment_data
    # 对给定的案例结果执行 moment 方法测试
    def test_moment_against_mp(self, case_result):
        # 从案例结果中获取源案例和 MP 结果
        src_case = case_result["src_case"]
        mp_result = case_result["mp_result"]
        # 提取源案例中的 m、k、v 值
        mkv = src_case["m"], src_case["k"], src_case["v"]

        # 忽略无效值遇到的警告。实际问题将通过结果比较捕获。
        with np.errstate(invalid='ignore'):
            # 调用 stats.studentized_range.moment 方法
            res = stats.studentized_range.moment(*mkv)

        # 断言结果接近于 MP 结果
        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],
                        rtol=src_case["expected_rtol"])

    @pytest.mark.slow
    # 执行 PDF 集成测试
    def test_pdf_integration(self):
        k, v = 3, 10
        # 测试 PDF 集成是否为 1
        res = quad(stats.studentized_range.pdf, 0, np.inf, args=(k, v))
        assert_allclose(res[0], 1)

    @pytest.mark.xslow
    # 测试 PDF 是否与 CDF 匹配
    def test_pdf_against_cdf(self):
        k, v = 3, 10

        # 使用累积梯形法集成 PDF，检查其是否与 CDF 匹配
        x = np.arange(0, 10, step=0.01)

        # 获取 CDF 和原始 PDF 数据
        y_cdf = stats.studentized_range.cdf(x, k, v)[1:]
        y_pdf_raw = stats.studentized_range.pdf(x, k, v)
        y_pdf_cumulative = cumulative_trapezoid(y_pdf_raw, x)

        # 由于累积误差，使用相对较大的 rtol 断言结果接近
        assert_allclose(y_pdf_cumulative, y_cdf, rtol=1e-4)

    @pytest.mark.parametrize("r_case_result", r_data)
    # 测试是否与 R 中的结果匹配
    def test_cdf_against_r(self, r_case_result):
        # 使用 R 的结果进行大 `v` 值测试
        q, k, v, r_res = r_case_result
        with np.errstate(invalid='ignore'):
            # 调用 stats.studentized_range.cdf 方法
            res = stats.studentized_range.cdf(q, k, v)
        # 断言结果接近于 R 的结果
        assert_allclose(res, r_res)

    @pytest.mark.xslow
    # 测试矩向量化
    def test_moment_vectorization(self):
        # 测试矩向量化。直接调用 `_munp` 因为 `rv_continuous.moment` 目前有问题。详见 gh-12192

        # 忽略无效值遇到的警告。实际问题将通过结果比较捕获。
        with np.errstate(invalid='ignore'):
            # 调用 stats.studentized_range._munp 方法
            m = stats.studentized_range._munp([1, 2], [4, 5], [10, 11])

        # 断言返回的形状为 (2,)
        assert_allclose(m.shape, (2,))

        # 使用 pytest.raises 断言是否会引发 ValueError，并匹配特定消息
        with pytest.raises(ValueError, match="...could not be broadcast..."):
            stats.studentized_range._munp(1, [4, 5], [10, 11, 12])

    @pytest.mark.xslow
    # 测试 fitstart 方法的有效性
    def test_fitstart_valid(self):
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            # 可能会有集成警告消息不同
            sup.filter(IntegrationWarning)
            # 调用 stats.studentized_range._fitstart 方法
            k, df, _, _ = stats.studentized_range._fitstart([1, 2, 3])
        # 断言参数检查结果为真
        assert_(stats.studentized_range._argcheck(k, df))
    def test_infinite_df(self):
        # 检查当自由度很高时，CDF和PDF的无限和正常积分器大致匹配的情况
        res = stats.studentized_range.pdf(3, 10, np.inf)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        # 使用 assert_allclose 断言检查两个结果是否接近
        assert_allclose(res, res_finite, atol=1e-4, rtol=1e-4)

        res = stats.studentized_range.cdf(3, 10, np.inf)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        # 使用 assert_allclose 断言检查两个结果是否接近
        assert_allclose(res, res_finite, atol=1e-4, rtol=1e-4)

    def test_df_cutoff(self):
        # 测试在自由度为100,000时，CDF和PDF是否正确地切换积分器。
        # 无限积分器应该足够不同，导致 allclose 断言失败。
        # 同时，通过使用相同积分器，确保在自由度相差1时通过 allclose 断言，这应该是非常小的差异。
        res = stats.studentized_range.pdf(3, 10, 100000)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        res_sanity = stats.studentized_range.pdf(3, 10, 99998)
        # 使用 assert_raises 检查是否会引发 AssertionError
        assert_raises(AssertionError, assert_allclose, res, res_finite,
                      atol=1e-6, rtol=1e-6)
        # 使用 assert_allclose 断言检查两个结果是否接近
        assert_allclose(res_finite, res_sanity, atol=1e-6, rtol=1e-6)

        res = stats.studentized_range.cdf(3, 10, 100000)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        res_sanity = stats.studentized_range.cdf(3, 10, 99998)
        # 使用 assert_raises 检查是否会引发 AssertionError
        assert_raises(AssertionError, assert_allclose, res, res_finite,
                      atol=1e-6, rtol=1e-6)
        # 使用 assert_allclose 断言检查两个结果是否接近
        assert_allclose(res_finite, res_sanity, atol=1e-6, rtol=1e-6)

    def test_clipping(self):
        # 在某些系统上，这个计算的结果是 -9.9253938401489e-14。
        # 正确的结果非常接近零，但不应为负数。
        q, k, v = 34.6413996195345746, 3, 339
        # 计算学生化范围的生存函数值
        p = stats.studentized_range.sf(q, k, v)
        # 使用 assert_allclose 断言检查结果是否接近零
        assert_allclose(p, 0, atol=1e-10)
        # 使用 assert 断言检查结果是否大于等于零
        assert p >= 0
def test_540_567():
    # 测试在票号 540 和 567 中返回 NaN 的情况
    assert_almost_equal(stats.norm.cdf(-1.7624320982), 0.03899815971089126,
                        decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(-1.7624320983), 0.038998159702449846,
                        decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(1.38629436112, loc=0.950273420309,
                                       scale=0.204423758009),
                        0.98353464004309321,
                        decimal=10, err_msg='test_540_567')


def test_regression_ticket_1326():
    # 调整以避免在 0*log(0) 时返回 NaN
    assert_almost_equal(stats.chi2.pdf(0.0, 2), 0.5, 14)


def test_regression_tukey_lambda():
    # 确保 Tukey-Lambda 分布正确处理非正的 lambda 值
    x = np.linspace(-5.0, 5.0, 101)

    with np.errstate(divide='ignore'):
        for lam in [0.0, -1.0, -2.0, np.array([[-1.0], [0.0], [-2.0]])]:
            p = stats.tukeylambda.pdf(x, lam)
            assert_((p != 0.0).all())
            assert_(~np.isnan(p).all())

        lam = np.array([[-1.0], [0.0], [2.0]])
        p = stats.tukeylambda.pdf(x, lam)

    assert_(~np.isnan(p).all())
    assert_((p[0] != 0.0).all())
    assert_((p[1] != 0.0).all())
    assert_((p[2] != 0.0).any())
    assert_((p[2] == 0.0).any())


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
def test_regression_ticket_1421():
    assert_('pdf(x, mu, loc=0, scale=1)' not in stats.poisson.__doc__)
    assert_('pmf(x,' in stats.poisson.__doc__)


def test_nan_arguments_gh_issue_1362():
    with np.errstate(invalid='ignore'):
        assert_(np.isnan(stats.t.logcdf(1, np.nan)))
        assert_(np.isnan(stats.t.cdf(1, np.nan)))
        assert_(np.isnan(stats.t.logsf(1, np.nan)))
        assert_(np.isnan(stats.t.sf(1, np.nan)))
        assert_(np.isnan(stats.t.pdf(1, np.nan)))
        assert_(np.isnan(stats.t.logpdf(1, np.nan)))
        assert_(np.isnan(stats.t.ppf(1, np.nan)))
        assert_(np.isnan(stats.t.isf(1, np.nan)))

        assert_(np.isnan(stats.bernoulli.logcdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.cdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logsf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.sf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.pmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logpmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.ppf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.isf(np.nan, 0.5)))


def test_frozen_fit_ticket_1536():
    np.random.seed(5678)
    true = np.array([0.25, 0., 0.5])
    x = stats.lognorm.rvs(true[0], true[1], true[2], size=100)

    with np.errstate(divide='ignore'):
        params = np.array(stats.lognorm.fit(x, floc=0.))

    assert_almost_equal(params, true, decimal=2)

    params = np.array(stats.lognorm.fit(x, fscale=0.5, loc=0))
    # 断言验证参数 `params` 是否与 `true` 几乎相等，精度为小数点后两位
    assert_almost_equal(params, true, decimal=2)
    
    # 使用数据 `x` 进行对数正态分布拟合，指定初始参数 `f0=0.25` 和 `loc=0`
    params = np.array(stats.lognorm.fit(x, f0=0.25, loc=0))
    # 断言验证参数 `params` 是否与 `true` 几乎相等，精度为小数点后两位
    assert_almost_equal(params, true, decimal=2)
    
    # 使用数据 `x` 进行对数正态分布拟合，指定初始参数 `f0=0.25` 和 `floc=0`
    params = np.array(stats.lognorm.fit(x, f0=0.25, floc=0))
    # 断言验证参数 `params` 是否与 `true` 几乎相等，精度为小数点后两位
    assert_almost_equal(params, true, decimal=2)
    
    # 设置随机数种子为 `5678`
    np.random.seed(5678)
    # 设置正态分布的均值 `loc` 为 1，指定初始参数 `floc=0.9`
    loc = 1
    floc = 0.9
    # 生成 100 个服从指定正态分布的随机数 `x`
    x = stats.norm.rvs(loc, 2., size=100)
    # 使用数据 `x` 进行正态分布拟合，指定初始参数 `floc=floc`
    params = np.array(stats.norm.fit(x, floc=floc))
    # 计算期望的参数值 `expected`，包括 `floc` 和标准差的估计
    expected = np.array([floc, np.sqrt(((x-floc)**2).mean())])
    # 断言验证参数 `params` 是否与期望的参数 `expected` 几乎相等，精度为小数点后四位
    assert_almost_equal(params, expected, decimal=4)
# 测试回归：检查柯西分布拟合的起始值是否有效。
def test_regression_ticket_1530():
    # 设定随机种子，确保结果可重现
    np.random.seed(654321)
    # 生成柯西分布的随机变量
    rvs = stats.cauchy.rvs(size=100)
    # 对随机变量进行柯西分布拟合，得到参数
    params = stats.cauchy.fit(rvs)
    # 预期的拟合参数
    expected = (0.045, 1.142)
    # 断言拟合参数接近预期值，精确度为小数点后一位
    assert_almost_equal(params, expected, decimal=1)


# 测试 GitHub PR #4806：检查柯西分布拟合的起始值是否有效。
def test_gh_pr_4806():
    # 设定随机种子，确保结果可重现
    np.random.seed(1234)
    # 生成标准正态分布的随机变量
    x = np.random.randn(42)
    # 对每个偏移量进行柯西分布拟合
    for offset in 10000.0, 1222333444.0:
        loc, scale = stats.cauchy.fit(x + offset)
        # 断言拟合后的位置参数接近偏移量，允许误差为1.0
        assert_allclose(loc, offset, atol=1.0)
        # 断言拟合后的尺度参数接近0.6，允许误差为1.0
        assert_allclose(scale, 0.6, atol=1.0)


# 测试 Tukey Lambda 分布的统计特性与 Ticket #1545 相关。
def test_tukeylambda_stats_ticket_1545():
    # 计算 Tukey Lambda 分布的一些统计量，包括均值、方差、偏度和峰度
    mv = stats.tukeylambda.stats(0, moments='mvsk')
    # 已知的精确值
    expected = [0, np.pi**2/3, 0, 1.2]
    # 断言计算得到的统计量接近预期值，精确度为小数点后十位
    assert_almost_equal(mv, expected, decimal=10)

    mv = stats.tukeylambda.stats(3.13, moments='mvsk')
    # 使用 mpmath 计算得到的精确值
    expected = [0, 0.0269220858861465102, 0, -0.898062386219224104]
    # 断言计算得到的统计量接近预期值，精确度为小数点后十位
    assert_almost_equal(mv, expected, decimal=10)

    mv = stats.tukeylambda.stats(0.14, moments='mvsk')
    # 使用 mpmath 计算得到的精确值
    expected = [0, 2.11029702221450250, 0, -0.02708377353223019456]
    # 断言计算得到的统计量接近预期值，精确度为小数点后十位
    assert_almost_equal(mv, expected, decimal=10)


# 测试 Poisson 分布的对数概率质量函数与 Ticket #1436 相关。
def test_poisson_logpmf_ticket_1436():
    # 断言计算得到的 Poisson 分布对数概率质量函数结果是有限的
    assert_(np.isfinite(stats.poisson.logpmf(1500, 200)))


# 测试 Powerlaw 统计函数。
def test_powerlaw_stats():
    """测试 Powerlaw 统计函数。

    该单元测试同时也是 Ticket #1548 的回归测试。

    精确的值为:
    均值:
        mu = a / (a + 1)
    方差:
        sigma**2 = a / ((a + 2) * (a + 1) ** 2)
    偏度:
        可以使用公式 gamma_1 = -2.0 * ((a - 1) / (a + 3)) * sqrt((a + 2) / a) 计算
    峰度:
        可以使用公式 gamma_2 = 6 * (a**3 - a**2 - 6*a + 2) / (a*(a+3)*(a+4)) 计算
    """
    cases = [(1.0, (0.5, 1./12, 0.0, -1.2)),
             (2.0, (2./3, 2./36, -0.56568542494924734, -0.6))]
    # 对于给定的每个参数 a 和其对应的准确 mvsk 统计量，执行以下操作：
    for a, exact_mvsk in cases:
        # 使用 powerlaw 模块计算给定参数 a 的 mvsk 统计量
        mvsk = stats.powerlaw.stats(a, moments="mvsk")
        # 断言计算得到的 mvsk 统计量与预期的 exact_mvsk 几乎相等
        assert_array_almost_equal(mvsk, exact_mvsk)
def test_powerlaw_edge():
    # 回归测试，用于修复问题 gh-3986。
    # 计算对数概率密度函数在 x=0 处的值
    p = stats.powerlaw.logpdf(0, 1)
    # 断言计算结果等于 0.0
    assert_equal(p, 0.0)


def test_exponpow_edge():
    # 回归测试，用于修复问题 gh-3982。
    # 计算指数幂分布的对数概率密度函数在 x=0 处的值
    p = stats.exponpow.logpdf(0, 1)
    # 断言计算结果等于 0.0
    assert_equal(p, 0.0)

    # 对于其他 b 值，检查 x=0 处的概率密度函数和对数概率密度函数
    p = stats.exponpow.pdf(0, [0.25, 1.0, 1.5])
    # 断言计算结果分别为无穷大、1.0、0.0
    assert_equal(p, [np.inf, 1.0, 0.0])
    p = stats.exponpow.logpdf(0, [0.25, 1.0, 1.5])
    # 断言计算结果分别为无穷大、0.0、负无穷
    assert_equal(p, [np.inf, 0.0, -np.inf])


def test_gengamma_edge():
    # 回归测试，用于修复问题 gh-3985。
    # 计算广义伽玛分布的概率密度函数在 x=0 处的值
    p = stats.gengamma.pdf(0, 1, 1)
    # 断言计算结果等于 1.0
    assert_equal(p, 1.0)


@pytest.mark.parametrize("a, c, ref, tol",
                         [(1500000.0, 1, 8.529426144018633, 1e-15),
                          (1e+30, 1, 35.95771492811536, 1e-15),
                          (1e+100, 1, 116.54819318290696, 1e-15),
                          (3e3, 1, 5.422011196659015, 1e-13),
                          (3e6, -1e100, -236.29663213396054, 1e-15),
                          (3e60, 1e-100, 1.3925371786831085e+102, 1e-15)])
def test_gengamma_extreme_entropy(a, c, ref, tol):
    # 参考值是使用 mpmath 计算得到的:
    # from mpmath import mp
    # mp.dps = 500
    #
    # def gen_entropy(a, c):
    #     a, c = mp.mpf(a), mp.mpf(c)
    #     val = mp.digamma(a)
    #     h = (a * (mp.one - val) + val/c + mp.loggamma(a) - mp.log(abs(c)))
    #     return float(h)
    # 断言广义伽玛分布的熵计算结果接近于参考值 ref
    assert_allclose(stats.gengamma.entropy(a, c), ref, rtol=tol)


def test_gengamma_endpoint_with_neg_c():
    # 回归测试，用于修复问题。
    # 当 c 为负数时，广义伽玛分布在 x=0 处的概率密度函数应该为 0.0
    p = stats.gengamma.pdf(0, 1, -1)
    assert p == 0.0
    # 当 c 为负数时，广义伽玛分布在 x=0 处的对数概率密度函数应该为负无穷
    logp = stats.gengamma.logpdf(0, 1, -1)
    assert logp == -np.inf


def test_gengamma_munp():
    # 回归测试，用于修复问题 gh-4724。
    # 计算广义伽玛分布的负阶矩 _munp(-2, a, c) 的值
    p = stats.gengamma._munp(-2, 200, 1.)
    # 断言计算结果接近于理论值 1./199/198
    assert_almost_equal(p, 1./199/198)

    p = stats.gengamma._munp(-2, 10, 1.)
    # 断言计算结果接近于理论值 1./9/8
    assert_almost_equal(p, 1./9/8)


def test_ksone_fit_freeze():
    # 回归测试，用于修复问题 ticket #1638。
    # 包含一组数据 d，用于拟合 ksone 分布
    d = np.array(
        [-0.18879233, 0.15734249, 0.18695107, 0.27908787, -0.248649,
         -0.2171497, 0.12233512, 0.15126419, 0.03119282, 0.4365294,
         0.08930393, -0.23509903, 0.28231224, -0.09974875, -0.25196048,
         0.11102028, 0.1427649, 0.10176452, 0.18754054, 0.25826724,
         0.05988819, 0.0531668, 0.21906056, 0.32106729, 0.2117662,
         0.10886442, 0.09375789, 0.24583286, -0.22968366, -0.07842391,
         -0.31195432, -0.21271196, 0.1114243, -0.13293002, 0.01331725,
         -0.04330977, -0.09485776, -0.28434547, 0.22245721, -0.18518199,
         -0.10943985, -0.35243174, 0.06897665, -0.03553363, -0.0701746,
         -0.06037974, 0.37670779, -0.21684405])
    # 进入一个上下文管理器，用于处理 NumPy 的错误状态
    with np.errstate(invalid='ignore'):
        # 进入另一个上下文管理器，用于抑制特定的警告信息
        with suppress_warnings() as sup:
            # 过滤掉积分警告，指定警告消息以避免显示
            sup.filter(IntegrationWarning,
                       "The maximum number of subdivisions .50. has been "
                       "achieved.")
            # 过滤掉运行时警告，指定警告消息以避免显示
            sup.filter(RuntimeWarning,
                       "floating point number truncated to an integer")
            # 对数据进行 KS One Sample 拟合
            stats.ksone.fit(d)
def test_norm_logcdf():
    # Test precision of the logcdf of the normal distribution.
    # This precision was enhanced in ticket 1614.
    x = -np.asarray(list(range(0, 120, 4)))
    # Values from R
    expected = [-0.69314718, -10.36010149, -35.01343716, -75.41067300,
                -131.69539607, -203.91715537, -292.09872100, -396.25241451,
                -516.38564863, -652.50322759, -804.60844201, -972.70364403,
                -1156.79057310, -1356.87055173, -1572.94460885, -1805.01356068,
                -2053.07806561, -2317.13866238, -2597.19579746, -2893.24984493,
                -3205.30112136, -3533.34989701, -3877.39640444, -4237.44084522,
                -4613.48339520, -5005.52420869, -5413.56342187, -5837.60115548,
                -6277.63751711, -6733.67260303]

    # Asserting the closeness of logcdf values with expected values
    assert_allclose(stats.norm().logcdf(x), expected, atol=1e-8)

    # also test the complex-valued code path
    assert_allclose(stats.norm().logcdf(x + 1e-14j).real, expected, atol=1e-8)

    # test the accuracy: d(logcdf)/dx = pdf / cdf \equiv exp(logpdf - logcdf)
    # Computing the derivative of logcdf with respect to x
    deriv = (stats.norm.logcdf(x + 1e-10j)/1e-10).imag
    deriv_expected = np.exp(stats.norm.logpdf(x) - stats.norm.logcdf(x))
    assert_allclose(deriv, deriv_expected, atol=1e-10)


def test_levy_cdf_ppf():
    # Test levy.cdf, including small arguments.
    x = np.array([1000, 1.0, 0.5, 0.1, 0.01, 0.001])

    # Expected values were calculated separately with mpmath.
    # E.g.
    # >>> mpmath.mp.dps = 100
    # >>> x = mpmath.mp.mpf('0.01')
    # >>> cdf = mpmath.erfc(mpmath.sqrt(1/(2*x)))
    expected = np.array([0.9747728793699604,
                         0.3173105078629141,
                         0.1572992070502851,
                         0.0015654022580025495,
                         1.523970604832105e-23,
                         1.795832784800726e-219])

    # Computing the cumulative distribution function (cdf) of Levy distribution
    y = stats.levy.cdf(x)
    assert_allclose(y, expected, rtol=1e-10)

    # ppf(expected) should get us back to x.
    # Computing the percent point function (inverse of cdf) of Levy distribution
    xx = stats.levy.ppf(expected)
    assert_allclose(xx, x, rtol=1e-13)


def test_levy_sf():
    # Large values, far into the tail of the distribution.
    x = np.array([1e15, 1e25, 1e35, 1e50])
    # Expected values were calculated with mpmath.
    expected = np.array([2.5231325220201597e-08,
                         2.52313252202016e-13,
                         2.52313252202016e-18,
                         7.978845608028653e-26])
    # Computing the survival function (1 - cdf) of Levy distribution
    y = stats.levy.sf(x)
    assert_allclose(y, expected, rtol=1e-14)


# The expected values for levy.isf(p) were calculated with mpmath.
# For loc=0 and scale=1, the inverse SF can be computed with
#
#     import mpmath
#
#     def levy_invsf(p):
#         return 1/(2*mpmath.erfinv(p)**2)
#
# For example, with mpmath.mp.dps set to 60, float(levy_invsf(1e-20))
# returns 6.366197723675814e+39.
#
@pytest.mark.parametrize('p, expected_isf',
                         [(1e-20, 6.366197723675814e+39),
                          (1e-8, 6366197723675813.0),
                          (0.375, 4.185810119346273),
                          (0.875, 0.42489442055310134),
                          (0.999, 0.09235685880262713),
                          (0.9999999962747097, 0.028766845244146945)])
def test_levy_isf(p, expected_isf):
    # 计算 Levy 分布的逆生存函数并验证结果
    x = stats.levy.isf(p)
    assert_allclose(x, expected_isf, atol=5e-15)


def test_levy_l_sf():
    # 测试 levy_l.sf 对于小参数的情况
    x = np.array([-0.016, -0.01, -0.005, -0.0015])
    # 预期值通过 mpmath 计算得到
    expected = np.array([2.6644463892359302e-15,
                         1.523970604832107e-23,
                         2.0884875837625492e-45,
                         5.302850374626878e-147])
    y = stats.levy_l.sf(x)
    assert_allclose(y, expected, rtol=1e-13)


def test_levy_l_isf():
    # 测试 roundtrip sf(isf(p))，包括一个小的输入值
    p = np.array([3.0e-15, 0.25, 0.99])
    x = stats.levy_l.isf(p)
    q = stats.levy_l.sf(x)
    assert_allclose(q, p, rtol=5e-14)


def test_hypergeom_interval_1802():
    # 这两个测试曾经导致无限循环
    assert_equal(stats.hypergeom.interval(.95, 187601, 43192, 757),
                 (152.0, 197.0))
    assert_equal(stats.hypergeom.interval(.945, 187601, 43192, 757),
                 (152.0, 197.0))
    # 这个测试以前也是正常工作的
    assert_equal(stats.hypergeom.interval(.94, 187601, 43192, 757),
                 (153.0, 196.0))

    # 边界情况 .a == .b
    assert_equal(stats.hypergeom.ppf(0.02, 100, 100, 8), 8)
    assert_equal(stats.hypergeom.ppf(1, 100, 100, 8), 8)


def test_distribution_too_many_args():
    np.random.seed(1234)

    # 检查当向方法传递了过多参数时是否会引发 TypeError
    # 这是针对 ticket 1815 的回归测试
    x = np.linspace(0.1, 0.7, num=5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, 5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.rvs, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.cdf, x, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.ppf, x, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.stats, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.entropy, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.fit, x, 2., 3, loc=1.0, scale=0.5)

    # 这些应该不会报错
    stats.gamma.pdf(x, 2, 3)  # loc=3
    stats.gamma.pdf(x, 2, 3, 4)  # loc=3, scale=4
    stats.gamma.stats(2., 3)
    stats.gamma.stats(2., 3, 4)
    stats.gamma.stats(2., 3, 4, 'mv')
    stats.gamma.rvs(2., 3, 4, 5)
    stats.gamma.fit(stats.gamma.rvs(2., size=7), 2.)
    # 计算几何分布的概率质量函数（PMF），给定参数 x, 2，和 loc=3
    stats.geom.pmf(x, 2, loc=3)  # no error, loc=3
    # 确保当传递额外参数时会抛出 TypeError 异常
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, 4)
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, loc=4)

    # 对指数分布和指数威布分布进行类似的异常测试，分别使用 3 个和 4 个参数，以及 loc=1.0
    assert_raises(TypeError, stats.expon.pdf, x, 3, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, 0.1, 0.1)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, loc=1.0)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, 1.0, scale=0.5)
    # 计算非中心 F 分布的概率密度函数（PDF），使用 3 个参数，加上 loc=1.0 和 scale=0.5
    stats.ncf.pdf(x, 3, 4, 5, 6, 1.0)  # 3 args, plus loc/scale


这段代码主要是对不同概率分布函数进行测试，确保它们在接收正确参数时不会出错，并且在传递额外或不正确的参数时会抛出 TypeError 异常。
def test_ncx2_tails_ticket_955():
    # Trac #955 -- check that the cdf computed by special functions
    # matches the integrated pdf
    # 使用特殊函数计算的累积分布函数（cdf）与积分后的概率密度函数（pdf）进行比较，以验证问题 #955
    a = stats.ncx2.cdf(np.arange(20, 25, 0.2), 2, 1.07458615e+02)
    # 计算 ncx2 分布的累积分布函数（cdf）
    b = stats.ncx2._cdfvec(np.arange(20, 25, 0.2), 2, 1.07458615e+02)
    # 使用内部函数 _cdfvec 计算 ncx2 分布的累积分布函数（cdf）
    assert_allclose(a, b, rtol=1e-3, atol=0)
    # 使用 numpy 的 assert_allclose 函数检查 a 和 b 是否在指定的误差范围内相等


def test_ncx2_tails_pdf():
    # ncx2.pdf does not return nans in extreme tails(example from gh-1577)
    # NB: this is to check that nan_to_num is not needed in ncx2.pdf
    # 检查 ncx2.pdf 在极端尾部不会返回 NaN（来自 gh-1577 的示例）
    # 注意：这是为了验证在 ncx2.pdf 中不需要 nan_to_num 函数
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # 使用警告过滤器捕获 RuntimeWarning 类型的警告
        assert_equal(stats.ncx2.pdf(1, np.arange(340, 350), 2), 0)
        # 验证给定参数下的 ncx2 概率密度函数（pdf）的返回值是否为 0
        logval = stats.ncx2.logpdf(1, np.arange(340, 350), 2)
        # 计算给定参数下的 ncx2 对数概率密度函数（logpdf）

    assert_(np.isneginf(logval).all())
    # 使用 numpy 的 assert_ 函数验证 logval 是否全部为负无穷

    # Verify logpdf has extended precision when pdf underflows to 0
    # 验证当 pdf 下溢至 0 时，logpdf 具有扩展精度
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # 使用警告过滤器捕获 RuntimeWarning 类型的警告
        assert_equal(stats.ncx2.pdf(10000, 3, 12), 0)
        # 验证给定参数下的 ncx2 概率密度函数（pdf）的返回值是否为 0
        assert_allclose(stats.ncx2.logpdf(10000, 3, 12), -4662.444377524883)
        # 使用 numpy 的 assert_allclose 函数验证给定参数下的 ncx2 对数概率密度函数（logpdf）是否接近预期值


@pytest.mark.parametrize('method, expected', [
    ('cdf', np.array([2.497951336e-09, 3.437288941e-10])),
    ('pdf', np.array([1.238579980e-07, 1.710041145e-08])),
    ('logpdf', np.array([-15.90413011, -17.88416331])),
    ('ppf', np.array([4.865182052, 7.017182271]))
])
def test_ncx2_zero_nc(method, expected):
    # gh-5441
    # ncx2 with nc=0 is identical to chi2
    # Comparison to R (v3.5.1)
    # > options(digits=10)
    # > pchisq(0.1, df=10, ncp=c(0,4))
    # > dchisq(0.1, df=10, ncp=c(0,4))
    # > dchisq(0.1, df=10, ncp=c(0,4), log=TRUE)
    # > qchisq(0.1, df=10, ncp=c(0,4))
    # 使用参数化测试方法，测试 ncx2 分布在 nc=0 时与 chi2 分布的等价性（gh-5441）
    # 与 R (v3.5.1) 的比较结果

    result = getattr(stats.ncx2, method)(0.1, nc=[0, 4], df=10)
    # 使用 getattr 函数动态获取 stats.ncx2 对象的方法并调用，测试不同方法的结果
    assert_allclose(result, expected, atol=1e-15)
    # 使用 numpy 的 assert_allclose 函数验证 result 和 expected 是否在指定的误差范围内相等


def test_ncx2_zero_nc_rvs():
    # gh-5441
    # ncx2 with nc=0 is identical to chi2
    # 测试 ncx2 分布在 nc=0 时与 chi2 分布的随机变量生成的等价性（gh-5441）
    result = stats.ncx2.rvs(df=10, nc=0, random_state=1)
    # 生成 ncx2 分布在给定参数下的随机变量
    expected = stats.chi2.rvs(df=10, random_state=1)
    # 生成 chi2 分布在给定参数下的随机变量
    assert_allclose(result, expected, atol=1e-15)
    # 使用 numpy 的 assert_allclose 函数验证 result 和 expected 是否在指定的误差范围内相等


def test_ncx2_gh12731():
    # test that gh-12731 is resolved; previously these were all 0.5
    # 测试 gh-12731 是否解决；先前这些值都是 0.5
    nc = 10**np.arange(5, 10)
    # 创建一个数组，用于测试 ncx2 分布的累积分布函数（cdf）
    assert_equal(stats.ncx2.cdf(1e4, df=1, nc=nc), 0)
    # 使用 assert_equal 函数验证给定参数下的 ncx2 分布的累积分布函数（cdf）是否为 0


def test_ncx2_gh8665():
    # test that gh-8665 is resolved; previously this tended to nonzero value
    # 测试 gh-8665 是否解决；先前这些值倾向于非零值
    x = np.array([4.99515382e+00, 1.07617327e+01, 2.31854502e+01,
                  4.99515382e+01, 1.07617327e+02, 2.31854502e+02,
                  4.99515382e+02, 1.07617327e+03, 2.31854502e+03,
                  4.99515382e+03, 1.07617327e+04, 2.31854502e+04])
    nu, lam = 20, 499.51538166556196

    sf = stats.ncx2.sf(x, df=nu, nc=lam)
    # 计算 ncx2 分布的生存函数（sf）
    # 以下是在 R 中计算的结果，找不到生存函数的实现
    # options(digits=16)
    # x <- c(4.99515382e+00, 1.07617327e+01, 2.31854502e+01, 4.99515382e+01,
    #        1.07617327e+02, 2.31854502e+02, 4.99515382e+02, 1.07617327e+03,
    #        2.31854502e+03, 4.99515382e+03, 1.07617327e+04, 2.31854502e+04,
    # 暂未完成 R 中生存函数的计算
    # 定义预期的生存函数（Survival Function）值列表，用于与计算结果比较
    sf_expected = [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                   1.0000000000000000, 1.0000000000000000, 0.9999999999999888,
                   0.6646525582135460, 0.0000000000000000, 0.0000000000000000,
                   0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                   0.0000000000000000]
    # 使用 assert_allclose 函数检查计算出的生存函数值 sf 是否与预期的 sf_expected 相近
    # 设置绝对容差 atol=1e-12，以确保计算的精度
    assert_allclose(sf, sf_expected, atol=1e-12)
# 定义一个测试函数，用于回归测试 GitHub 问题 #11777
def test_ncx2_gh11777():
    # 设置自由度 df 为 6700
    df = 6700
    # 设置非中心参数 nc 为 5300
    nc = 5300
    # 在 ncx2 分布中生成 x 值，范围从边界百分点到百分点，总数为 10000
    x = np.linspace(stats.ncx2.ppf(0.001, df, nc),
                    stats.ncx2.ppf(0.999, df, nc), num=10000)
    # 计算 ncx2 分布的概率密度函数
    ncx2_pdf = stats.ncx2.pdf(x, df, nc)
    # 使用正态分布的概率密度函数作为高斯近似
    gauss_approx = stats.norm.pdf(x, df + nc, np.sqrt(2 * df + 4 * nc))
    # 由于只寻找明显的差异，使用大容差
    # 断言 ncx2_pdf 与 gauss_approx 的近似度在容差 1e-4 内
    assert_allclose(ncx2_pdf, gauss_approx, atol=1e-4)


# 预期值是使用 mpmath 计算 foldnorm.sf 的结果：
#
#    from mpmath import mp
#    mp.dps = 60
#    def foldcauchy_sf(x, c):
#        x = mp.mpf(x)
#        c = mp.mpf(c)
#        return mp.one - (mp.atan(x - c) + mp.atan(x + c))/mp.pi
#
# 例如：
#
#    >>> float(foldcauchy_sf(2, 1))
#    0.35241638234956674
#
@pytest.mark.parametrize('x, c, expected',
                         [(2, 1, 0.35241638234956674),
                          (2, 2, 0.5779791303773694),
                          (1e13, 1, 6.366197723675813e-14),
                          (2e16, 1, 3.183098861837907e-17),
                          (1e13, 2e11, 6.368745221764519e-14),
                          (0.125, 200, 0.999998010612169)])
def test_foldcauchy_sf(x, c, expected):
    # 计算 foldcauchy.sf 的值
    sf = stats.foldcauchy.sf(x, c)
    # 断言计算出的 sf 与预期值 expected 在容差 2e-15 内近似
    assert_allclose(sf, expected, 2e-15)


# 与上述 test_foldcauchy_sf() 中的 mpmath 代码相同，
# 用于创建 test_halfcauchy_sf() 中的预期值。
@pytest.mark.parametrize('x, expected',
                         [(2, 0.2951672353008665),
                          (1e13, 6.366197723675813e-14),
                          (2e16, 3.183098861837907e-17),
                          (5e80, 1.2732395447351629e-81)])
def test_halfcauchy_sf(x, expected):
    # 计算 halfcauchy.sf 的值
    sf = stats.halfcauchy.sf(x)
    # 断言计算出的 sf 与预期值 expected 在容差 2e-15 内近似
    assert_allclose(sf, expected, 2e-15)


# 预期值是使用 mpmath 计算的：
#     expected = mp.cot(mp.pi*p/2)
@pytest.mark.parametrize('p, expected',
                         [(0.9999995, 7.853981633329977e-07),
                          (0.975, 0.039290107007669675),
                          (0.5, 1.0),
                          (0.01, 63.65674116287158),
                          (1e-14, 63661977236758.13),
                          (5e-80, 1.2732395447351627e+79)])
def test_halfcauchy_isf(p, expected):
    # 计算 halfcauchy.isf 的值
    x = stats.halfcauchy.isf(p)
    # 断言计算出的 x 与预期值 expected 近似
    assert_allclose(x, expected)


def test_foldnorm_zero():
    # 参数值 c=0 未启用，请参见 GitHub 问题 #2399。
    # 创建一个 foldnorm 分布的随机变量 rv，参数 scale=1
    rv = stats.foldnorm(0, scale=1)
    # 断言 rv 的累积分布函数在 0 处的值为 0，此前 rv.cdf(0) 的结果为 nan
    assert_equal(rv.cdf(0), 0)


# 预期值是使用 mpmath 计算 foldnorm.sf 的结果：
#
#    from mpmath import mp
#    mp.dps = 60
#    def foldnorm_sf(x, c):
#        x = mp.mpf(x)
#        c = mp.mpf(c)
#        return mp.ncdf(-x+c) + mp.ncdf(-x-c)
#
# 例如：
#
#    >>> float(foldnorm_sf(2, 1))
#    0.16000515196308715
#
# 使用 pytest 的 parametrize 装饰器，定义了多组输入参数 (x, c, expected)，并对 test_foldnorm_sf 函数进行参数化测试
@pytest.mark.parametrize('x, c, expected',
                         [(2, 1, 0.16000515196308715),
                          (20, 1, 8.527223952630977e-81),
                          (10, 15, 0.9999997133484281),
                          (25, 15, 7.619853024160525e-24)])
def test_foldnorm_sf(x, c, expected):
    # 调用 scipy.stats 中的 foldnorm 分布的 sf 方法，计算生存函数值
    sf = stats.foldnorm.sf(x, c)
    # 使用 numpy.testing.assert_allclose 函数断言 sf 的值与期望值 expected 在误差范围 1e-14 内相等
    assert_allclose(sf, expected, 1e-14)


def test_stats_shapes_argcheck():
    # 对 stats.invgamma.stats 方法进行测试，传入不合法的参数 [0.0, 0.5, 1.0]，期望抛出异常
    # 此处 0 不是合法的参数 `a`
    mv3 = stats.invgamma.stats([0.0, 0.5, 1.0], 1, 0.5)
    # 传入合法的参数 [0.5, 1.0]，计算对应的统计量
    mv2 = stats.invgamma.stats([0.5, 1.0], 1, 0.5)
    # 将 mv2 中的每个元素与 NaN 构成的数组拼接，用于与 mv3 比较
    mv2_augmented = tuple(np.r_[np.nan, _] for _ in mv2)
    # 使用 numpy.testing.assert_equal 函数断言 mv2_augmented 与 mv3 相等
    assert_equal(mv2_augmented, mv3)

    # 传入不合法的参数 [-1]，期望抛出异常
    # -1 不是合法的形状参数
    mv3 = stats.lognorm.stats([2, 2.4, -1])
    # 传入合法的参数 [2, 2.4]，计算对应的统计量
    mv2 = stats.lognorm.stats([2, 2.4])
    # 将 mv2 中的每个元素与 NaN 构成的数组拼接，用于与 mv3 比较
    mv2_augmented = tuple(np.r_[_, np.nan] for _ in mv2)
    # 使用 numpy.testing.assert_equal 函数断言 mv2_augmented 与 mv3 相等
    assert_equal(mv2_augmented, mv3)

    # FIXME: 这只是一个快速粗糙的 bug 修复测试。
    # 对于具有多个形状参数的 stats 方法，向量化可能不正确，因此某些分布可能会失败。


# 测试通过显式形状参数子类化分布

class _distr_gen(stats.rv_continuous):
    # 定义一个子类，重写 _pdf 方法
    def _pdf(self, x, a):
        return 42


class _distr2_gen(stats.rv_continuous):
    # 定义另一个子类，重写 _cdf 方法
    def _cdf(self, x, a):
        return 42 * a + x


class _distr3_gen(stats.rv_continuous):
    # 定义第三个子类，重写 _pdf 方法，接受额外参数 b
    def _pdf(self, x, a, b):
        return a + b

    # _cdf 方法与 _pdf 方法的形状参数不一致，用于检查检查程序是否能捕获这种不一致性。
    def _cdf(self, x, a):
        return 42 * a + x


class _distr6_gen(stats.rv_continuous):
    # 定义第四个子类，具有两个形状参数（_pdf 和 _cdf 方法均定义，形状一致）
    def _pdf(self, x, a, b):
        return a*x + b

    def _cdf(self, x, a, b):
        return 42 * a + x


class TestSubclassingExplicitShapes:
    # 构造一个具有显式形状参数的分布并进行测试

    def test_correct_shapes(self):
        # 创建一个名为 dummy、形状参数为 'a' 的 _distr_gen 分布实例
        dummy_distr = _distr_gen(name='dummy', shapes='a')
        # 断言调用 dummy_distr 实例的 pdf 方法，传入参数 1，得到结果与期望值 42 相等
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_wrong_shapes_1(self):
        # 创建一个名为 dummy、不合法形状参数为 'A' 的 _distr_gen 分布实例，期望抛出 TypeError 异常
        dummy_distr = _distr_gen(name='dummy', shapes='A')
        # 使用 assert_raises 函数断言 dummy_distr.pdf 方法调用时抛出 TypeError 异常
        assert_raises(TypeError, dummy_distr.pdf, 1, **dict(a=1))

    def test_wrong_shapes_2(self):
        # 创建一个名为 dummy、不合法形状参数为 'a, b, c' 的 _distr_gen 分布实例，期望抛出 TypeError 异常
        dummy_distr = _distr_gen(name='dummy', shapes='a, b, c')
        # 创建一个包含形状参数 a=1, b=2, c=3 的字典 dct
        dct = dict(a=1, b=2, c=3)
        # 使用 assert_raises 函数断言 dummy_distr.pdf 方法调用时抛出 TypeError 异常
        assert_raises(TypeError, dummy_distr.pdf, 1, **dct)

    def test_shapes_string(self):
        # 创建一个包含形状参数值不是字符串的字典 dct
        dct = dict(name='dummy', shapes=42)
        # 使用 assert_raises 函数断言创建 _distr_gen 实例时抛出 TypeError 异常
        assert_raises(TypeError, _distr_gen, **dct)

    def test_shapes_identifiers_1(self):
        # 创建一个名为 dummy、形状参数为 '(!)' 的 _distr_gen 分布实例，期望抛出 SyntaxError 异常
        dct = dict(name='dummy', shapes='(!)')
        # 使用 assert_raises 函数断言创建 _distr_gen 实例时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)
    def test_shapes_identifiers_2(self):
        # 创建包含 'name' 和 'shapes' 键值对的字典
        dct = dict(name='dummy', shapes='4chan')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_3(self):
        # 创建包含 'name' 和 'shapes' 键值对的字典
        dct = dict(name='dummy', shapes='m(fti)')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_nodefaults(self):
        # 创建包含 'name' 和 'shapes' 键值对的字典
        dct = dict(name='dummy', shapes='a=2')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_args(self):
        # 创建包含 'name' 和 'shapes' 键值对的字典
        dct = dict(name='dummy', shapes='*args')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_kwargs(self):
        # 创建包含 'name' 和 'shapes' 键值对的字典
        dct = dict(name='dummy', shapes='**kwargs')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_keywords(self):
        # 不允许使用 Python 关键字作为形状参数
        dct = dict(name='dummy', shapes='a, b, c, lambda')
        # 断言调用 _distr_gen 函数时抛出 SyntaxError 异常
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_signature(self):
        # 测试与 _pdf 签名匹配的显式形状
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a

        # 创建 _dist_gen 实例，并调用 pdf 方法验证结果
        dist = _dist_gen(shapes='a')
        assert_equal(dist.pdf(0.5, a=2), stats.norm.pdf(0.5)*2)

    def test_shapes_signature_inconsistent(self):
        # 测试与 _pdf 签名不匹配的显式形状
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a

        # 创建 _dist_gen 实例，预期调用 pdf 方法时抛出 TypeError 异常
        dist = _dist_gen(shapes='a, b')
        assert_raises(TypeError, dist.pdf, 0.5, **dict(a=1, b=2))

    def test_star_args(self):
        # 测试只有 starargs 的 _pdf 方法
        # 注意：pdf 方法的 **kwargs 永远不会传递给 _pdf 方法
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg

        # 创建 _dist_gen 实例，调用 pdf 方法验证结果
        dist = _dist_gen(shapes='extra_kwarg')
        assert_equal(dist.pdf(0.5, extra_kwarg=33), stats.norm.pdf(0.5)*33)
        assert_equal(dist.pdf(0.5, 33), stats.norm.pdf(0.5)*33)
        assert_raises(TypeError, dist.pdf, 0.5, **dict(xxx=33))

    def test_star_args_2(self):
        # 测试带有命名参数和 starargs 的 _pdf 方法
        # 注意：pdf 方法的 **kwargs 永远不会传递给 _pdf 方法
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, offset, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg + offset

        # 创建 _dist_gen 实例，调用 pdf 方法验证结果
        dist = _dist_gen(shapes='offset, extra_kwarg')
        assert_equal(dist.pdf(0.5, offset=111, extra_kwarg=33),
                     stats.norm.pdf(0.5)*33 + 111)
        assert_equal(dist.pdf(0.5, 111, 33),
                     stats.norm.pdf(0.5)*33 + 111)
    def test_extra_kwarg(self):
        # **kwargs to _pdf are ignored.
        # 框架限制，_pdf(x, *goodargs)只接受*args参数。
        class _distr_gen(stats.rv_continuous):
            def _pdf(self, x, *args, **kwargs):
                # _pdf应当自行处理*args和**kwargs。这里的处理方式是忽略*args，
                # 并查找``extra_kwarg``并使用它。
                extra_kwarg = kwargs.pop('extra_kwarg', 1)
                return stats.norm._pdf(x) * extra_kwarg

        # 创建_distr_gen对象，传入'shapes'='extra_kwarg'作为参数
        dist = _distr_gen(shapes='extra_kwarg')
        # 断言dist.pdf(1, extra_kwarg=3)等于stats.norm.pdf(1)
        assert_equal(dist.pdf(1, extra_kwarg=3), stats.norm.pdf(1))

    def test_shapes_empty_string(self):
        # shapes=''等同于shapes=None
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x):
                return stats.norm.pdf(x)

        # 创建_dist_gen对象，传入shapes=''作为参数
        dist = _dist_gen(shapes='')
        # 断言dist.pdf(0.5)等于stats.norm.pdf(0.5)
        assert_equal(dist.pdf(0.5), stats.norm.pdf(0.5))
class TestSubclassingNoShapes:
    # 构建一个没有显式形状参数的分布对象，并对其进行测试。

    def test_only__pdf(self):
        # 使用 _distr_gen 创建一个名为 dummy 的分布对象，并断言其 pdf 方法的返回值为 42
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_only__cdf(self):
        # 使用 _distr2_gen 创建一个名为 dummy 的分布对象，并对其 pdf 方法的返回值进行准确度断言
        # 这里假设 _pdf 是通过对 _cdf 进行数值导数计算得到的
        dummy_distr = _distr2_gen(name='dummy')
        assert_almost_equal(dummy_distr.pdf(1, a=1), 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
    def test_signature_inspection(self):
        # 检查 _pdf 的签名检查是否正常工作，并且是否在类的文档字符串中使用了该信息
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 1)
        assert_equal(dummy_distr.shapes, 'a')
        res = re.findall(r'logpdf\(x, a, loc=0, scale=1\)',
                         dummy_distr.__doc__)
        assert_(len(res) == 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
    def test_signature_inspection_2args(self):
        # 对于有两个形状参数并且定义了 _pdf 和 _cdf 的情况，进行相同的检查
        dummy_distr = _distr6_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 2)
        assert_equal(dummy_distr.shapes, 'a, b')
        res = re.findall(r'logpdf\(x, a, b, loc=0, scale=1\)',
                         dummy_distr.__doc__)
        assert_(len(res) == 1)

    def test_signature_inspection_2args_incorrect_shapes(self):
        # 定义了 _pdf 和 _cdf，但形状不一致的情况下应该引发 TypeError
        assert_raises(TypeError, _distr3_gen, name='dummy')

    def test_defaults_raise(self):
        # 默认参数应该引发异常
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a=42):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_starargs_raise(self):
        # 没有显式形状参数时，*args 不允许使用
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a, *args):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_kwargs_raise(self):
        # 没有显式形状参数时，**kwargs 不允许使用
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a, **kwargs):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
def test_docstrings():
    # 检查统计模块中所有分布类的文档字符串是否包含指定的错误模式
    badones = [r',\s*,', r'\(\s*,', r'^\s*:']
    for distname in stats.__all__:
        dist = getattr(stats, distname)
        if isinstance(dist, (stats.rv_discrete, stats.rv_continuous)):
            for regex in badones:
                assert_(re.search(regex, dist.__doc__) is None)


def test_infinite_input():
    # 断言 skellam 分布的 sf 方法在输入无穷大时的返回值接近于 0
    assert_almost_equal(stats.skellam.sf(np.inf, 10, 11), 0)
    # 断言 ncx2 分布的 _cdf 方法在输入无穷大时的返回值接近于 1
    assert_almost_equal(stats.ncx2._cdf(np.inf, 8, 0.1), 1)


def test_lomax_accuracy():
    # 留待后续完善
    # 进行 gh-4033 的回归测试
    p = stats.lomax.ppf(stats.lomax.cdf(1e-100, 1), 1)
    # 断言 p 的值接近 1e-100
    assert_allclose(p, 1e-100)
def test_truncexpon_accuracy():
    # 对 gh-4035 的回归测试
    p = stats.truncexpon.ppf(stats.truncexpon.cdf(1e-100, 1), 1)
    # 断言 p 的值接近 1e-100
    assert_allclose(p, 1e-100)


def test_rayleigh_accuracy():
    # 对 gh-4034 的回归测试
    p = stats.rayleigh.isf(stats.rayleigh.sf(9, 1), 1)
    # 断言 p 的值几乎等于 9.0，精确度为小数点后 15 位
    assert_almost_equal(p, 9.0, decimal=15)


def test_genextreme_give_no_warnings():
    """gh-6219 的回归测试，验证不产生警告"""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # 以下是对 stats.genextreme 的各函数调用，以验证不产生警告
        stats.genextreme.cdf(.5, 0)
        stats.genextreme.pdf(.5, 0)
        stats.genextreme.ppf(.5, 0)
        stats.genextreme.logpdf(-np.inf, 0.0)
        
        # 获取警告的数量
        number_of_warnings_thrown = len(w)
        # 断言没有产生警告
        assert_equal(number_of_warnings_thrown, 0)


def test_genextreme_entropy():
    # 对 gh-5181 的回归测试，验证 stats.genextreme 的熵函数
    euler_gamma = 0.5772156649015329

    h = stats.genextreme.entropy(-1.0)
    # 断言 h 的值接近 2 * 欧拉常数 + 1，相对误差小于等于 1e-14
    assert_allclose(h, 2*euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(0)
    # 断言 h 的值接近 欧拉常数 + 1，相对误差小于等于 1e-14
    assert_allclose(h, euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(1.0)
    # 断言 h 的值等于 1
    assert_equal(h, 1)

    h = stats.genextreme.entropy(-2.0, scale=10)
    # 断言 h 的值接近 3 * 欧拉常数 + ln(10) + 1，相对误差小于等于 1e-14
    assert_allclose(h, euler_gamma*3 + np.log(10) + 1, rtol=1e-14)

    h = stats.genextreme.entropy(10)
    # 断言 h 的值接近 -9 * 欧拉常数 + 1，相对误差小于等于 1e-14
    assert_allclose(h, -9*euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(-10)
    # 断言 h 的值接近 11 * 欧拉常数 + 1，相对误差小于等于 1e-14
    assert_allclose(h, 11*euler_gamma + 1, rtol=1e-14)


def test_genextreme_sf_isf():
    # 预期值使用 mpmath 计算：
    #
    #    import mpmath
    #
    #    def mp_genextreme_sf(x, xi, mu=0, sigma=1):
    #        # 使用维基百科的公式，此处 xi 的符号约定与 scipy 的形状参数相反。
    #        if xi != 0:
    #            t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    #        else:
    #            t = mpmath.exp(-(x - mu)/sigma)
    #        return 1 - mpmath.exp(-t)
    #
    # >>> mpmath.mp.dps = 1000
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("1e8"), mpmath.mp.mpf("0.125"))
    # >>> float(s)
    # 1.6777205262585625e-57
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("7.98"), mpmath.mp.mpf("-0.125"))
    # >>> float(s)
    # 1.52587890625e-21
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("7.98"), mpmath.mp.mpf("0"))
    # >>> float(s)
    # 0.00034218086528426593

    x = 1e8
    s = stats.genextreme.sf(x, -0.125)
    # 断言 s 的值接近 1.6777205262585625e-57
    assert_allclose(s, 1.6777205262585625e-57)
    x2 = stats.genextreme.isf(s, -0.125)
    # 断言 x2 的值接近 x，即 1e8
    assert_allclose(x2, x)

    x = 7.98
    s = stats.genextreme.sf(x, 0.125)
    # 断言 s 的值接近 1.52587890625e-21
    assert_allclose(s, 1.52587890625e-21)
    x2 = stats.genextreme.isf(s, 0.125)
    # 断言 x2 的值接近 x，即 7.98
    assert_allclose(x2, x)

    x = 7.98
    s = stats.genextreme.sf(x, 0)
    # 断言 s 的值接近 0.00034218086528426593
    assert_allclose(s, 0.00034218086528426593)
    x2 = stats.genextreme.isf(s, 0)
    # 断言 x2 的值接近 x，即 7.98
    assert_allclose(x2, x)


def test_burr12_ppf_small_arg():
    prob = 1e-16
    quantile = stats.burr12.ppf(prob, 2, 3)
    # 预期的分位数使用 mpmath 计算：
    #   >>> import mpmath
    ```
    # 设置多精度数学库 mpmath 的精度为 100 位
    >>> mpmath.mp.dps = 100
    # 定义概率 prob 为 1e-16，使用 mpmath.mpf 将其转换为多精度浮点数
    >>> prob = mpmath.mpf('1e-16')
    # 定义常数 c 和 d 为 2 和 3，使用 mpmath.mpf 将它们转换为多精度浮点数
    >>> c = mpmath.mpf(2)
    >>> d = mpmath.mpf(3)
    # 计算表达式 float(((1-prob)**(-1/d) - 1)**(1/c)) 的值，得到结果约为 5.7735026918962575e-09
    >>> float(((1-prob)**(-1/d) - 1)**(1/c))
    5.7735026918962575e-09
    # 使用 assert_allclose 函数验证 quantile 是否接近预期值 5.7735026918962575e-09
    assert_allclose(quantile, 5.7735026918962575e-09)
def test_crystalball_function():
    """
    All values are calculated using the independent implementation of the
    ROOT framework (see https://root.cern.ch/).
    Corresponding ROOT code is given in the comments.
    """
    # 生成从 -5.0 到 5.0 的等间隔的数组，步长为0.5，并且排除最后一个元素
    X = np.linspace(-5.0, 5.0, 21)[:-1]

    # 使用 crystalball 概率密度函数计算概率密度
    calculated = stats.crystalball.pdf(X, beta=1.0, m=2.0)
    # 预期结果
    expected = np.array([0.0202867, 0.0241428, 0.0292128, 0.0360652, 0.045645,
                         0.059618, 0.0811467, 0.116851, 0.18258, 0.265652,
                         0.301023, 0.265652, 0.18258, 0.097728, 0.0407391,
                         0.013226, 0.00334407, 0.000658486, 0.000100982,
                         1.20606e-05])
    # 检查计算结果与预期结果的接近程度
    assert_allclose(expected, calculated, rtol=0.001)

    # 使用 crystalball 概率密度函数计算概率密度，不同参数
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0)
    # 预期结果
    expected = np.array([0.0019648, 0.00279754, 0.00417592, 0.00663121,
                         0.0114587, 0.0223803, 0.0530497, 0.12726, 0.237752,
                         0.345928, 0.391987, 0.345928, 0.237752, 0.12726,
                         0.0530497, 0.0172227, 0.00435458, 0.000857469,
                         0.000131497, 1.57051e-05])
    # 检查计算结果与预期结果的接近程度
    assert_allclose(expected, calculated, rtol=0.001)

    # 使用 crystalball 概率密度函数计算概率密度，包括额外的 loc 和 scale 参数
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    # 预期结果
    expected = np.array([0.00785921, 0.0111902, 0.0167037, 0.0265249,
                         0.0423866, 0.0636298, 0.0897324, 0.118876, 0.147944,
                         0.172964, 0.189964, 0.195994, 0.189964, 0.172964,
                         0.147944, 0.118876, 0.0897324, 0.0636298, 0.0423866,
                         0.0265249])
    # 检查计算结果与预期结果的接近程度
    assert_allclose(expected, calculated, rtol=0.001)

    # 使用 crystalball 累积分布函数计算累积概率密度
    calculated = stats.crystalball.cdf(X, beta=1.0, m=2.0)
    # 预期结果
    expected = np.array([0.12172, 0.132785, 0.146064, 0.162293, 0.18258,
                         0.208663, 0.24344, 0.292128, 0.36516, 0.478254,
                         0.622723, 0.767192, 0.880286, 0.94959, 0.982834,
                         0.995314, 0.998981, 0.999824, 0.999976, 0.999997])
    # 检查计算结果与预期结果的接近程度
    assert_allclose(expected, calculated, rtol=0.001)

    # 使用 crystalball 累积分布函数计算累积概率密度，不同参数
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0)
    # 期望值是一个 NumPy 数组，表示预期的概率密度函数的值
    expected = np.array([0.00442081, 0.00559509, 0.00730787, 0.00994682,
                         0.0143234, 0.0223803, 0.0397873, 0.0830763, 0.173323,
                         0.320592, 0.508717, 0.696841, 0.844111, 0.934357,
                         0.977646, 0.993899, 0.998674, 0.999771, 0.999969,
                         0.999997])
    # 使用 assert_allclose 函数检查计算出的值和期望值之间的近似程度
    assert_allclose(expected, calculated, rtol=0.001)

    # 使用 crystalball.cdf 计算 Crystal Ball 分布的累积分布函数值
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    # 期望值是一个 NumPy 数组，表示预期的累积分布函数的值
    expected = np.array([0.0176832, 0.0223803, 0.0292315, 0.0397873, 0.0567945,
                         0.0830763, 0.121242, 0.173323, 0.24011, 0.320592,
                         0.411731, 0.508717, 0.605702, 0.696841, 0.777324,
                         0.844111, 0.896192, 0.934357, 0.960639, 0.977646])
    # 使用 assert_allclose 函数检查计算出的值和期望值之间的近似程度
    assert_allclose(expected, calculated, rtol=0.001)
def test_crystalball_function_moments():
    """
    All values are calculated using the pdf formula and the integrate function
    of Mathematica
    """
    # 定义 beta 和 m 数组，用于测试不同参数组合的 Crystalball 分布
    beta = np.array([2.0, 1.0, 3.0, 2.0, 3.0])
    m = np.array([3.0, 3.0, 2.0, 4.0, 9.0])

    # 断言 0 阶矩的计算结果与期望值接近
    expected_0th_moment = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    calculated_0th_moment = stats.crystalball._munp(0, beta, m)
    assert_allclose(expected_0th_moment, calculated_0th_moment, rtol=0.001)

    # 1 阶矩计算，与 WolframAlpha 的结果进行比较
    norm = np.array([2.5511, 3.01873, 2.51065, 2.53983, 2.507410455])
    a = np.array([-0.21992, -3.03265, np.inf, -0.135335, -0.003174])
    expected_1th_moment = a / norm
    calculated_1th_moment = stats.crystalball._munp(1, beta, m)
    assert_allclose(expected_1th_moment, calculated_1th_moment, rtol=0.001)

    # 2 阶矩计算，与预期值进行比较
    a = np.array([np.inf, np.inf, np.inf, 3.2616, 2.519908])
    expected_2th_moment = a / norm
    calculated_2th_moment = stats.crystalball._munp(2, beta, m)
    assert_allclose(expected_2th_moment, calculated_2th_moment, rtol=0.001)

    # 3 阶矩计算，与预期值进行比较
    a = np.array([np.inf, np.inf, np.inf, np.inf, -0.0577668])
    expected_3th_moment = a / norm
    calculated_3th_moment = stats.crystalball._munp(3, beta, m)
    assert_allclose(expected_3th_moment, calculated_3th_moment, rtol=0.001)

    # 4 阶矩计算，与预期值进行比较
    a = np.array([np.inf, np.inf, np.inf, np.inf, 7.78468])
    expected_4th_moment = a / norm
    calculated_4th_moment = stats.crystalball._munp(4, beta, m)
    assert_allclose(expected_4th_moment, calculated_4th_moment, rtol=0.001)

    # 5 阶矩计算，与预期值进行比较
    a = np.array([np.inf, np.inf, np.inf, np.inf, -1.31086])
    expected_5th_moment = a / norm
    calculated_5th_moment = stats.crystalball._munp(5, beta, m)
    assert_allclose(expected_5th_moment, calculated_5th_moment, rtol=0.001)


def test_crystalball_entropy():
    # 回归测试 gh-13602
    cb = stats.crystalball(2, 3)
    res1 = cb.entropy()
    
    # 使用梯形法计算熵的数值积分，与 Crystalball 的 PDF 相比较
    lo, hi, N = -20000, 30, 200000
    x = np.linspace(lo, hi, N)
    res2 = trapezoid(entr(cb.pdf(x)), x)
    assert_allclose(res1, res2, rtol=1e-7)


def test_invweibull_fit():
    """
    Test fitting invweibull to data.

    Here is a the same calculation in R:

    > library(evd)
    > library(fitdistrplus)
    > x = c(1, 1.25, 2, 2.5, 2.8,  3, 3.8, 4, 5, 8, 10, 12, 64, 99)
    > result = fitdist(x, 'frechet', control=list(reltol=1e-13),
    +                  fix.arg=list(loc=0), start=list(shape=2, scale=3))
    > result
    Fitting of the distribution ' frechet ' by maximum likelihood
    Parameters:
          estimate Std. Error
    shape 1.048482  0.2261815
    scale 3.099456  0.8292887
    """
    Fixed parameters:
        value
    loc     0

    """

    # 定义优化器函数，使用 scipy.optimize.fmin 进行优化
    def optimizer(func, x0, args=(), disp=0):
        return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)

    # 给定数据数组 x
    x = np.array([1, 1.25, 2, 2.5, 2.8, 3, 3.8, 4, 5, 8, 10, 12, 64, 99])
    # 拟合逆威布尔分布的参数 c, loc, scale，其中 loc 固定为 0，使用自定义的优化器 optimizer
    c, loc, scale = stats.invweibull.fit(x, floc=0, optimizer=optimizer)
    # 断言检查拟合得到的参数 c 是否接近给定值 1.048482
    assert_allclose(c, 1.048482, rtol=5e-6)
    # 断言检查拟合得到的 loc 参数是否为 0
    assert loc == 0
    # 断言检查拟合得到的参数 scale 是否接近给定值 3.099456
    assert_allclose(scale, 3.099456, rtol=5e-6)
# 使用 pytest 模块的 @pytest.mark.parametrize 装饰器，对 test_invweibull_sf 函数进行参数化测试。
# 参数化的参数包括 x, c 和 expected，分别是输入值，形状参数和预期输出值。
@pytest.mark.parametrize('x, c, expected',
                         [(3, 1.5, 0.175064510070713299327),
                          (2000, 1.5, 1.11802773877318715787e-5),
                          (2000, 9.25, 2.92060308832269637092e-31),
                          (1e15, 1.5, 3.16227766016837933199884e-23)])
def test_invweibull_sf(x, c, expected):
    # 计算逆威布尔分布的生存函数（Survival function），即 1 - CDF(x, c)
    computed = stats.invweibull.sf(x, c)
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(computed, expected, rtol=1e-15)


# 使用 pytest 模块的 @pytest.mark.parametrize 装饰器，对 test_invweibull_isf 函数进行参数化测试。
# 参数化的参数包括 p, c 和 expected，分别是概率值，形状参数和预期输出值。
@pytest.mark.parametrize('p, c, expected',
                         [(0.5, 2.5, 1.15789669836468183976),
                          (3e-18, 5, 3195.77171838060906447)])
def test_invweibull_isf(p, c, expected):
    # 计算逆威布尔分布的逆生存函数（Inverse survival function），即 对应于 p 的 x 值
    computed = stats.invweibull.isf(p, c)
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(computed, expected, rtol=1e-15)


# 使用 pytest.mark.parametrize 装饰器，对 test_ncf_edge_case 函数进行参数化测试。
# 参数化的参数包括 df1, df2 和 x，分别是自由度，自由度，和输入数组。
@pytest.mark.parametrize(
    'df1,df2,x',
    [(2, 2, [-0.5, 0.2, 1.0, 2.3]),
     (4, 11, [-0.5, 0.2, 1.0, 2.3]),
     (7, 17, [1, 2, 3, 4, 5])]
)
def test_ncf_edge_case(df1, df2, x):
    # 测试 GitHub 问题编号 gh-11660 中描述的边缘情况。
    # 当 nc = 0 时，非中心 F 分布应与 F 分布相同。
    nc = 0
    # 使用 stats.f.cdf 计算 F 分布的累积分布函数（CDF）
    expected_cdf = stats.f.cdf(x, df1, df2)
    # 使用 stats.ncf.cdf 计算非中心 F 分布的累积分布函数（CDF）
    calculated_cdf = stats.ncf.cdf(x, df1, df2, nc)
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(expected_cdf, calculated_cdf, rtol=1e-14)

    # 当 ncf_gen._skip_pdf 被用于替代通用的概率密度函数（PDF）时，
    # 这个额外的测试将会很有用。
    # 使用 stats.f.pdf 计算 F 分布的概率密度函数（PDF）
    expected_pdf = stats.f.pdf(x, df1, df2)
    # 使用 stats.ncf.pdf 计算非中心 F 分布的概率密度函数（PDF）
    calculated_pdf = stats.ncf.pdf(x, df1, df2, nc)
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(expected_pdf, calculated_pdf, rtol=1e-6)


def test_ncf_variance():
    # 回归测试 GitHub 问题编号 gh-10658（非中心 F 分布的方差计算错误）。
    # stats.ncf.var(2, 6, 4) 的正确值是 42.75，可以通过 Wolfram Alpha 或 Boost 库进行验证。
    v = stats.ncf.var(2, 6, 4)
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(v, 42.75, rtol=1e-14)


def test_ncf_cdf_spotcheck():
    # 回归测试 GitHub 问题编号 gh-15582，针对 R/MATLAB 中的值进行验证。
    # 从 R 或 MATLAB 中生成 check_val 的值。
    # R: pf(20, df1 = 6, df2 = 33, ncp = 30.4) = 0.998921
    # MATLAB: ncfcdf(20, 6, 33, 30.4) = 0.998921
    scipy_val = stats.ncf.cdf(20, 6, 33, 30.4)
    check_val = 0.998921
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(check_val, np.round(scipy_val, decimals=6))


# 使用 pytest.mark.skipif 装饰器，当系统位数小于等于 32 位时跳过测试。
@pytest.mark.skipif(sys.maxsize <= 2**32,
                    reason="On some 32-bit the warning is not raised")
def test_ncf_ppf_issue_17026():
    # 回归测试 GitHub 问题编号 gh-17026。
    # 创建包含 600 个元素的等差数列 x，其中第一个元素设置为 1e-16。
    x = np.linspace(0, 1, 600)
    x[0] = 1e-16
    par = (0.1, 2, 5, 0, 1)
    # 使用 pytest.warns 检查运行时警告，确保警告被正确地抛出。
    with pytest.warns(RuntimeWarning):
        q = stats.ncf.ppf(x, *par)
        q0 = [stats.ncf.ppf(xi, *par) for xi in x]
    # 使用 assert_allclose 函数比较计算结果与预期结果的接近程度
    assert_allclose(q, q0)


# 定义一个测试类 TestHistogram，用于编写关于直方图的测试。
class TestHistogram:
    def setup_method(self):
        # 设置随机种子为1234，确保结果可重复
        np.random.seed(1234)

        # 我们有8个区间
        # [1,2), [2,3), [3,4), [4,5), [5,6), [6,7), [7,8), [8,9)
        # 但实际上 np.histogram 会将最后的9放入 [8,9) 区间！
        # 因此，下面的最后一个区间与你预期的有轻微差异。
        # 构建直方图，用于创建连续分布的模板
        histogram = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                                  6, 6, 6, 6, 7, 7, 7, 8, 8, 9], bins=8)
        self.template = stats.rv_histogram(histogram)

        # 生成正态分布数据，作为正态分布模板
        data = stats.norm.rvs(loc=1.0, scale=2.5, size=10000, random_state=123)
        norm_histogram = np.histogram(data, bins=50)
        self.norm_template = stats.rv_histogram(norm_histogram)

    def test_pdf(self):
        # 测试数值
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        # 预期的概率密度函数值
        pdf_values = np.asarray([0.0/25.0, 0.0/25.0, 1.0/25.0, 1.0/25.0,
                                 2.0/25.0, 2.0/25.0, 3.0/25.0, 3.0/25.0,
                                 4.0/25.0, 4.0/25.0, 5.0/25.0, 5.0/25.0,
                                 4.0/25.0, 4.0/25.0, 3.0/25.0, 3.0/25.0,
                                 3.0/25.0, 3.0/25.0, 0.0/25.0, 0.0/25.0])
        # 断言连续分布的 pdf 方法计算结果与预期值相近
        assert_allclose(self.template.pdf(values), pdf_values)

        # 明确测试边界情况：
        # 如上所述，区间 [8,9) 的 pdf 比预期大，因为 np.histogram 将 9 放入了 [8,9) 区间。
        assert_almost_equal(self.template.pdf(8.0), 3.0/25.0)
        assert_almost_equal(self.template.pdf(8.5), 3.0/25.0)
        # 9 超出了我们定义的区间 [8,9)，因此 pdf 应为 0
        # 对于连续分布，这是正常的，因为单个值没有有限的概率！
        assert_almost_equal(self.template.pdf(9.0), 0.0/25.0)
        assert_almost_equal(self.template.pdf(10.0), 0.0/25.0)

        # 生成一个区间为 [-2, 2] 的数组 x，并断言正态分布模板的 pdf 与 scipy.stats.norm 的计算结果相近
        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.pdf(x),
                        stats.norm.pdf(x, loc=1.0, scale=2.5), rtol=0.1)
    # 测试累积分布函数（CDF）和反函数（PPF）的正确性
    def test_cdf_ppf(self):
        # 创建包含测试值的 NumPy 数组
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        # 创建包含预期累积分布函数值的 NumPy 数组
        cdf_values = np.asarray([0.0/25.0, 0.0/25.0, 0.0/25.0, 0.5/25.0,
                                 1.0/25.0, 2.0/25.0, 3.0/25.0, 4.5/25.0,
                                 6.0/25.0, 8.0/25.0, 10.0/25.0, 12.5/25.0,
                                 15.0/25.0, 17.0/25.0, 19.0/25.0, 20.5/25.0,
                                 22.0/25.0, 23.5/25.0, 25.0/25.0, 25.0/25.0])
        # 断言累积分布函数的计算结果与预期的值相近
        assert_allclose(self.template.cdf(values), cdf_values)
        
        # 断言反函数在给定累积分布函数值范围内的计算结果与预期的值相近
        assert_allclose(self.template.ppf(cdf_values[2:-1]), values[2:-1])

        # 测试累积分布函数和反函数是否互为反函数
        x = np.linspace(1.0, 9.0, 100)
        assert_allclose(self.template.ppf(self.template.cdf(x)), x)
        x = np.linspace(0.0, 1.0, 100)
        assert_allclose(self.template.cdf(self.template.ppf(x)), x)

        # 使用不同的参数范围测试正态分布的累积分布函数
        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.cdf(x),
                        stats.norm.cdf(x, loc=1.0, scale=2.5), rtol=0.1)

    # 测试随机变量生成函数的正确性
    def test_rvs(self):
        N = 10000
        # 生成随机样本
        sample = self.template.rvs(size=N, random_state=123)
        # 断言样本中小于 1.0 的数量为零
        assert_equal(np.sum(sample < 1.0), 0.0)
        # 断言样本中小于等于 2.0 的数量与预期相近
        assert_allclose(np.sum(sample <= 2.0), 1.0/25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 2.5), 2.0/25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 3.0), 3.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 3.5), 4.5/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.0), 6.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.5), 8.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 5.0), 10.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 5.5), 12.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.0), 15.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.5), 17.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.0), 19.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.5), 20.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.0), 22.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.5), 23.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05)
        # 断言样本中大于 9.0 的数量为零
        assert_equal(np.sum(sample > 9.0), 0.0)

    # 测试原点矩函数（mu_n_prime）的正确性
    def test_munp(self):
        # 遍历不同的阶数 n
        for n in range(4):
            # 断言原点矩函数的计算结果与对应正态分布的原点矩相近
            assert_allclose(self.norm_template._munp(n),
                            stats.norm(1.0, 2.5).moment(n), rtol=0.05)

    # 测试熵函数的正确性
    def test_entropy(self):
        # 断言熵函数的计算结果与对应正态分布的熵相近
        assert_allclose(self.norm_template.entropy(),
                        stats.norm.entropy(loc=1.0, scale=2.5), rtol=0.05)
def test_histogram_non_uniform():
    # Tests rv_histogram works even for non-uniform bin widths

    # 定义非均匀 bin 宽度的直方图 counts 和 bins
    counts, bins = ([1, 1], [0, 1, 1001])

    # 使用 counts 和 bins 创建一个离散分布 dist，density=False 表示概率密度函数未归一化
    dist = stats.rv_histogram((counts, bins), density=False)
    
    # 验证在指定点处的概率密度函数值是否接近给定值
    np.testing.assert_allclose(dist.pdf([0.5, 200]), [0.5, 0.0005])
    
    # 验证分布的中位数是否等于 1
    assert dist.median() == 1

    # 使用 counts 和 bins 创建一个离散分布 dist，density=True 表示概率密度函数已归一化
    dist = stats.rv_histogram((counts, bins), density=True)
    
    # 验证在指定点处的概率密度函数值是否接近给定值
    np.testing.assert_allclose(dist.pdf([0.5, 200]), 1/1001)
    
    # 验证分布的中位数是否等于 1001/2
    assert dist.median() == 1001/2

    # 检查省略 density 参数是否会对非均匀 bin 提示警告
    message = "Bin widths are not constant. Assuming..."
    with pytest.warns(RuntimeWarning, match=message):
        # 创建离散分布 dist，省略 density 参数，默认行为类似于 density=True
        dist = stats.rv_histogram((counts, bins))
        # 验证分布的中位数是否等于 1001/2
        assert dist.median() == 1001/2

    # 对于均匀 bin，不应提示警告
    dist = stats.rv_histogram((counts, [0, 1, 2]))
    assert dist.median() == 1


class TestLogUniform:
    def test_alias(self):
        # This test makes sure that "reciprocal" and "loguniform" are
        # aliases of the same distribution and that both are log-uniform

        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(98643218961)
        
        # 创建一个 loguniform 分布 rv
        rv = stats.loguniform(10 ** -3, 10 ** 0)
        
        # 从 rv 中生成随机样本 rvs
        rvs = rv.rvs(size=10000, random_state=rng)

        # 创建一个相同参数的 reciprocal 分布 rv2
        rng = np.random.default_rng(98643218961)
        rv2 = stats.reciprocal(10 ** -3, 10 ** 0)
        
        # 从 rv2 中生成随机样本 rvs2
        rvs2 = rv2.rvs(size=10000, random_state=rng)

        # 验证两个分布生成的随机样本是否接近
        assert_allclose(rvs2, rvs)

        # 对 rv 生成的随机样本取对数后，进行直方图统计并验证特定条件
        vals, _ = np.histogram(np.log10(rvs), bins=10)
        assert 900 <= vals.min() <= vals.max() <= 1100
        assert np.abs(np.median(vals) - 1000) <= 10

    @pytest.mark.parametrize("method", ['mle', 'mm'])
    def test_fit_override(self, method):
        # loguniform is overparameterized, so check that fit override enforces
        # scale=1 unless fscale is provided by the user

        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(98643218961)
        
        # 从 loguniform 分布生成随机样本 rvs
        rvs = stats.loguniform.rvs(0.1, 1, size=1000, random_state=rng)

        # 使用不同的拟合方法 method 对 rvs 进行参数拟合
        a, b, loc, scale = stats.loguniform.fit(rvs, method=method)
        
        # 验证是否默认 scale 等于 1
        assert scale == 1

        # 使用用户指定的 fscale 对 rvs 进行参数拟合
        a, b, loc, scale = stats.loguniform.fit(rvs, fscale=2, method=method)
        
        # 验证是否 scale 等于 2
        assert scale == 2

    def test_overflow(self):
        # original formulation had overflow issues; check that this is resolved
        # Extensive accuracy tests elsewhere, no need to test all methods

        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(7136519550773909093)
        
        # 设置 loguniform 分布的参数范围 a, b
        a, b = 1e-200, 1e200
        
        # 创建 loguniform 分布 dist
        dist = stats.loguniform(a, b)

        # 测试 roundtrip 错误
        cdf = rng.uniform(0, 1, size=1000)
        assert_allclose(dist.cdf(dist.ppf(cdf)), cdf)
        
        # 从 dist 中生成随机样本 rvs
        rvs = dist.rvs(size=1000)
        assert_allclose(dist.ppf(dist.cdf(rvs)), rvs)

        # 测试 pdf 的一个属性（确保没有溢出）
        x = 10.**np.arange(-200, 200)
        pdf = dist.pdf(x)  # no overflow
        assert_allclose(pdf[:-1]/pdf[1:], 10)

        # 检查均值 mean 是否与参考值相近
        mean = (b - a)/(np.log(b) - np.log(a))
        assert_allclose(dist.mean(), mean)


class TestArgus:
    def test_argus_rvs_large_chi(self):
        # 测试算法能否处理较大的 chi 值
        x = stats.argus.rvs(50, size=500, random_state=325)
        # 断言实际生成的随机变量均值与理论均值的接近程度
        assert_almost_equal(stats.argus(50).mean(), x.mean(), decimal=4)

    @pytest.mark.parametrize('chi, random_state', [
            [0.1, 325],   # chi <= 0.5: 拒绝法的第一种情况
            [1.3, 155],   # 0.5 < chi <= 1.8: 拒绝法的第二种情况
            [3.5, 135]    # chi > 1.8: 条件Gamma分布转换法
        ])
    def test_rvs(self, chi, random_state):
        # 生成指定 chi 值下的随机变量，并进行 Kolmogorov-Smirnov 检验
        x = stats.argus.rvs(chi, size=500, random_state=random_state)
        _, p = stats.kstest(x, "argus", (chi, ))
        # 断言 KS 检验的 p 值大于0.05
        assert_(p > 0.05)

    @pytest.mark.parametrize('chi', [1e-9, 1e-6])
    def test_rvs_small_chi(self, chi):
        # 测试当 chi 很小（接近0）时的情况，特别是 chi=0 的情况
        # 分布函数的 CDF 对于 chi=0 是 1 - (1 - x**2)**(3/2)
        # 测试生成的随机变量与 chi=0 时的分布的对比
        r = stats.argus.rvs(chi, size=500, random_state=890981)
        _, p = stats.kstest(r, lambda x: 1 - (1 - x**2)**(3/2))
        # 断言 KS 检验的 p 值大于0.05
        assert_(p > 0.05)

    # 期望值是使用 mpmath 计算得到的
    @pytest.mark.parametrize('chi, expected_mean',
                             [(1, 0.6187026683551835),
                              (10, 0.984805536783744),
                              (40, 0.9990617659702923),
                              (60, 0.9995831885165300),
                              (99, 0.9998469348663028)])
    def test_mean(self, chi, expected_mean):
        # 测试 Argus 分布的均值计算是否准确
        m = stats.argus.mean(chi, scale=1)
        # 断言计算得到的均值与期望值的接近程度
        assert_allclose(m, expected_mean, rtol=1e-13)

    # 期望值是使用 mpmath 计算得到的
    @pytest.mark.parametrize('chi, expected_var, rtol',
                             [(1, 0.05215651254197807, 1e-13),
                              (10, 0.00015805472008165595, 1e-11),
                              (40, 5.877763210262901e-07, 1e-8),
                              (60, 1.1590179389611416e-07, 1e-8),
                              (99, 1.5623277006064666e-08, 1e-8)])
    def test_var(self, chi, expected_var, rtol):
        # 测试 Argus 分布的方差计算是否准确
        v = stats.argus.var(chi, scale=1)
        # 断言计算得到的方差与期望值的接近程度
        assert_allclose(v, expected_var, rtol=rtol)

    # 期望值是使用 mpmath 计算得到的（参考 gh-13370）
    @pytest.mark.parametrize('chi, expected, rtol',
                             [(0.9, 0.07646314974436118, 1e-14),
                              (0.5, 0.015429797891863365, 1e-14),
                              (0.1, 0.0001325825293278049, 1e-14),
                              (0.01, 1.3297677078224565e-07, 1e-15),
                              (1e-3, 1.3298072023958999e-10, 1e-14),
                              (1e-4, 1.3298075973486862e-13, 1e-14),
                              (1e-6, 1.32980760133771e-19, 1e-14),
                              (1e-9, 1.329807601338109e-28, 1e-15)])
    def test_cdf(self, chi, expected, rtol):
        # 测试 Argus 分布的累积分布函数计算是否准确
        cdf = lambda x: 1 - (1 - x**2)**(3/2)
        _, p = stats.kstest(stats.argus.rvs(chi, size=500, random_state=890981), cdf)
        # 断言 KS 检验的 p 值大于0.05
        assert_(p > 0.05)
    def test_argus_phi_small_chi(self, chi, expected, rtol):
        # 使用 assert_allclose 函数检查 _argus_phi 函数的输出是否与期望值 expected 接近
        assert_allclose(_argus_phi(chi), expected, rtol=rtol)

    # 使用 pytest.mark.parametrize 标记进行参数化测试，测试 argus.pdf 函数
    # 预期值是通过 mpmath 计算得到的
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.28414073302940573, 1.2742227939992954, 1.2381254688255896)),
         (0.2, (0.296172952995264, 1.2951290588110516, 1.1865767100877576)),
         (0.1, (0.29791447523536274, 1.29806307956989, 1.1793168289857412)),
         (0.01, (0.2984904104866452, 1.2990283628160553, 1.1769268414080531)),
         (1e-3, (0.298496172925224, 1.2990380082487925, 1.176902956021053)),
         (1e-4, (0.29849623054991836, 1.2990381047023793, 1.1769027171686324)),
         (1e-6, (0.2984962311319278, 1.2990381056765605, 1.1769027147562232)),
         (1e-9, (0.298496231131986, 1.299038105676658, 1.1769027147559818))])
    def test_pdf_small_chi(self, chi, expected):
        # 创建输入数组 x
        x = np.array([0.1, 0.5, 0.9])
        # 使用 assert_allclose 函数检查 stats.argus.pdf 函数的输出是否与期望值 expected 接近
        assert_allclose(stats.argus.pdf(x, chi), expected, rtol=1e-13)

    # 使用 pytest.mark.parametrize 标记进行参数化测试，测试 argus.sf 函数
    # 预期值是通过 mpmath 计算得到的
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.9857660526895221, 0.6616565930168475, 0.08796070398429937)),
         (0.2, (0.9851555052359501, 0.6514666238985464, 0.08362690023746594)),
         (0.1, (0.9850670974995661, 0.6500061310508574, 0.08302050640683846)),
         (0.01, (0.9850378582451867, 0.6495239242251358, 0.08282109244852445)),
         (1e-3, (0.9850375656906663, 0.6495191015522573, 0.08281910005231098)),
         (1e-4, (0.9850375627651049, 0.6495190533254682, 0.08281908012852317)),
         (1e-6, (0.9850375627355568, 0.6495190528383777, 0.08281907992729293)),
         (1e-9, (0.9850375627355538, 0.649519052838329, 0.0828190799272728))])
    def test_sf_small_chi(self, chi, expected):
        # 创建输入数组 x
        x = np.array([0.1, 0.5, 0.9])
        # 使用 assert_allclose 函数检查 stats.argus.sf 函数的输出是否与期望值 expected 接近
        assert_allclose(stats.argus.sf(x, chi), expected, rtol=1e-14)

    # 使用 pytest.mark.parametrize 标记进行参数化测试，测试 argus.cdf 函数
    # 预期值是通过 mpmath 计算得到的
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.0142339473104779, 0.3383434069831524, 0.9120392960157007)),
         (0.2, (0.014844494764049919, 0.34853337610145363, 0.916373099762534)),
         (0.1, (0.014932902500433911, 0.34999386894914264, 0.9169794935931616)),
         (0.01, (0.014962141754813293, 0.35047607577486417, 0.9171789075514756)),
         (1e-3, (0.01496243430933372, 0.35048089844774266, 0.917180899947689)),
         (1e-4, (0.014962437234895118, 0.3504809466745317, 0.9171809198714769)),
         (1e-6, (0.01496243726444329, 0.3504809471616223, 0.9171809200727071)),
         (1e-9, (0.014962437264446245, 0.350480947161671, 0.9171809200727272))])
    def test_cdf_small_chi(self, chi, expected):
        # 创建输入数组 x
        x = np.array([0.1, 0.5, 0.9])
        # 使用 assert_allclose 函数检查 stats.argus.cdf 函数的输出是否与期望值 expected 接近
        assert_allclose(stats.argus.cdf(x, chi), expected, rtol=1e-12)

    # Expected values were computed with mpmath (code: see gh-13370).
    # 这是一个空注释，标明接下来的测试预期值是使用 mpmath 计算得到的，但实际上没有提供具体测试内容
    # 使用 pytest 的 @parametrize 装饰器，为 test_stats_small_chi 方法提供多组参数化输入
    @pytest.mark.parametrize(
        'chi, expected, rtol',
        [(0.5, (0.5964284712757741, 0.052890651988588604), 1e-12),
         (0.101, (0.5893490968089076, 0.053017469847275685), 1e-11),
         (0.1, (0.5893431757009437, 0.05301755449499372), 1e-13),
         (0.01, (0.5890515677940915, 0.05302167905837031), 1e-13),
         (1e-3, (0.5890486520005177, 0.053021719862088104), 1e-13),
         (1e-4, (0.5890486228426105, 0.0530217202700811), 1e-13),
         (1e-6, (0.5890486225481156, 0.05302172027420182), 1e-13),
         (1e-9, (0.5890486225480862, 0.05302172027420224), 1e-13)])
    # 测试方法，用于测试统计模块中 argus.stats 方法对给定参数的返回值
    def test_stats_small_chi(self, chi, expected, rtol):
        # 调用 argus.stats 方法获取返回值
        val = stats.argus.stats(chi, moments='mv')
        # 使用 assert_allclose 断言方法验证返回值是否与期望值接近
        assert_allclose(val, expected, rtol=rtol)
class TestNakagami:

    def test_logpdf(self):
        # 测试 Nakagami 分布的对数概率密度函数，对于一个小于 64 位浮点数表示范围的输入值。
        # 预期的 logpdf 值是用 mpmath 计算得出的:
        #
        #   def logpdf(x, nu):
        #       x = mpmath.mpf(x)
        #       nu = mpmath.mpf(nu)
        #       return (mpmath.log(2) + nu*mpmath.log(nu) -
        #               mpmath.loggamma(nu) + (2*nu - 1)*mpmath.log(x) -
        #               nu*x**2)
        #
        nu = 2.5
        x = 25
        logp = stats.nakagami.logpdf(x, nu)
        assert_allclose(logp, -1546.9253055607549)

    def test_sf_isf(self):
        # 测试 Nakagami 分布的生存函数 sf 和逆生存函数 isf，当生存函数值非常小时。
        # 预期的生存函数值是用 mpmath 计算得出的:
        #
        #   def sf(x, nu):
        #       x = mpmath.mpf(x)
        #       nu = mpmath.mpf(nu)
        #       return mpmath.gammainc(nu, nu*x*x, regularized=True)
        #
        nu = 2.5
        x0 = 5.0
        sf = stats.nakagami.sf(x0, nu)
        assert_allclose(sf, 2.736273158588307e-25, rtol=1e-13)
        # 检查逆操作是否回到 x0。
        x1 = stats.nakagami.isf(sf, nu)
        assert_allclose(x1, x0, rtol=1e-13)

    @pytest.mark.parametrize("m, ref",
        [(5, -0.097341814372152),
        (0.5, 0.7257913526447274),
        (10, -0.43426184310934907)])
    def test_entropy(self, m, ref):
        # 使用 sympy 和 mpmath 计算 Nakagami 分布的熵。
        # 这里显示了用于计算熵的 sympy 和 mpmath 代码，但并未实际执行。
        # 使用 mpmath 设置精度，计算熵并与参考值 ref 进行比较。
        assert_allclose(stats.nakagami.entropy(m), ref, rtol=1.1e-14)

    @pytest.mark.parametrize("m, ref",
        [(1e-100, -5.0e+99), (1e-10, -4999999965.442979),
         (9.999e6, -7.333206478668433), (1.001e7, -7.3337562313259825),
         (1e10, -10.787134112333835), (1e100, -114.40346329705756)])
    def test_extreme_nu(self, m, ref):
        # 测试极端参数 nu 下的 Nakagami 分布的熵。
        assert_allclose(stats.nakagami.entropy(m), ref)

    def test_entropy_overflow(self):
        # 检查在极端情况下计算熵是否仍然有限。
        assert np.isfinite(stats.nakagami._entropy(1e100))
        assert np.isfinite(stats.nakagami._entropy(1e-100))

    @pytest.mark.parametrize("nu, ref",
                             [(1e10, 0.9999999999875),
                              (1e3, 0.9998750078173821),
                              (1e-10, 1.772453850659802e-05)])
    def test_mean(self, nu, ref):
        # 计算均值的测试函数
        # 参考值是使用mpmath计算得出的
        # from mpmath import mp
        # mp.dps = 500
        # nu = mp.mpf(1e10)
        # float(mp.rf(nu, mp.mpf(0.5))/mp.sqrt(nu))
        assert_allclose(stats.nakagami.mean(nu), ref, rtol=1e-12)

    @pytest.mark.xfail(reason="Fit of nakagami not reliable, see gh-10908.")
    @pytest.mark.parametrize('nu', [1.6, 2.5, 3.9])
    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit(self, nu, loc, scale):
        # 针对nakagami分布拟合的回归测试（21/27案例之前失败）
        # 第一个参数值元组在gh-10908中讨论过
        N = 100
        samples = stats.nakagami.rvs(size=N, nu=nu, loc=loc,
                                     scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples)
        assert_allclose(nu_est, nu, rtol=0.2)
        assert_allclose(loc_est, loc, rtol=0.2)
        assert_allclose(scale_est, scale, rtol=0.2)

        def dlogl_dnu(nu, loc, scale):
            # 关于nu的对数似然梯度函数
            return ((-2*nu + 1) * np.sum(1/(samples - loc))
                    + 2*nu/scale**2 * np.sum(samples - loc))

        def dlogl_dloc(nu, loc, scale):
            # 关于loc的对数似然梯度函数
            return (N * (1 + np.log(nu) - polygamma(0, nu)) +
                    2 * np.sum(np.log((samples - loc) / scale))
                    - np.sum(((samples - loc) / scale)**2))

        def dlogl_dscale(nu, loc, scale):
            # 关于scale的对数似然梯度函数
            return (- 2 * N * nu / scale
                    + 2 * nu / scale ** 3 * np.sum((samples - loc) ** 2))

        assert_allclose(dlogl_dnu(nu_est, loc_est, scale_est), 0, atol=1e-3)
        assert_allclose(dlogl_dloc(nu_est, loc_est, scale_est), 0, atol=1e-3)
        assert_allclose(dlogl_dscale(nu_est, loc_est, scale_est), 0, atol=1e-3)

    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit_nu(self, loc, scale):
        # 对于nu = 0.5，我们有loc和scale的最大似然估计的解析值
        nu = 0.5
        n = 100
        samples = stats.nakagami.rvs(size=n, nu=nu, loc=loc,
                                     scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples, f0=nu)

        # 解析值
        loc_theo = np.min(samples)
        scale_theo = np.sqrt(np.mean((samples - loc_est) ** 2))

        assert_allclose(nu_est, nu, rtol=1e-7)
        assert_allclose(loc_est, loc_theo, rtol=1e-7)
        assert_allclose(scale_est, scale_theo, rtol=1e-7)
class TestWrapCauchy:

    def test_cdf_shape_broadcasting(self):
        # 回归测试 gh-13791.
        # 检查 wrapcauchy.cdf 是否正确地广播了形状参数。
        c = np.array([[0.03, 0.25], [0.5, 0.75]])
        x = np.array([[1.0], [4.0]])
        p = stats.wrapcauchy.cdf(x, c)
        assert p.shape == (2, 2)
        # 使用迭代器遍历 x 和 c 的每个元素，计算对应的标量值
        scalar_values = [stats.wrapcauchy.cdf(x1, c1)
                         for (x1, c1) in np.nditer((x, c))]
        assert_allclose(p.ravel(), scalar_values, rtol=1e-13)

    def test_cdf_center(self):
        # 测试 wrapcauchy.cdf 在中心点处的取值
        p = stats.wrapcauchy.cdf(np.pi, 0.03)
        assert_allclose(p, 0.5, rtol=1e-14)

    def test_cdf(self):
        x1 = 1.0  # 小于 pi 的值
        x2 = 4.0  # 大于 pi 的值
        c = 0.75
        # 测试 wrapcauchy.cdf 在不同输入下的表现
        p = stats.wrapcauchy.cdf([x1, x2], c)
        cr = (1 + c)/(1 - c)
        assert_allclose(p[0], np.arctan(cr*np.tan(x1/2))/np.pi)
        assert_allclose(p[1], 1 - np.arctan(cr*np.tan(np.pi - x2/2))/np.pi)


def test_rvs_no_size_error():
    # _rvs 方法必须有 size 参数；参见 gh-11394
    class rvs_no_size_gen(stats.rv_continuous):
        def _rvs(self):
            return 1

    rvs_no_size = rvs_no_size_gen(name='rvs_no_size')

    # 确保 _rvs 方法在调用时会抛出 TypeError 异常
    with assert_raises(TypeError, match=r"_rvs\(\) got (an|\d) unexpected"):
        rvs_no_size.rvs()


@pytest.mark.parametrize('distname, args', invdistdiscrete + invdistcont)
def test_support_gh13294_regression(distname, args):
    if distname in skip_test_support_gh13294_regression:
        pytest.skip(f"skipping test for the support method for "
                    f"distribution {distname}.")
    dist = getattr(stats, distname)
    # 使用无效参数测试支持方法
    if isinstance(dist, stats.rv_continuous):
        # 使用有效的 scale 进行测试
        if len(args) != 0:
            a0, b0 = dist.support(*args)
            assert_equal(a0, np.nan)
            assert_equal(b0, np.nan)
        # 使用无效的 scale 进行测试
        # 对于一些不需要参数的分布，只有无效的 scale 情况发生，
        # 因此在这个测试用例中隐式测试这种情况。
        loc1, scale1 = 0, -1
        a1, b1 = dist.support(*args, loc1, scale1)
        assert_equal(a1, np.nan)
        assert_equal(b1, np.nan)
    else:
        a, b = dist.support(*args)
        assert_equal(a, np.nan)
        assert_equal(b, np.nan)


def test_support_broadcasting_gh13294_regression():
    a0, b0 = stats.norm.support([0, 0, 0, 1], [1, 1, 1, -1])
    ex_a0 = np.array([-np.inf, -np.inf, -np.inf, np.nan])
    ex_b0 = np.array([np.inf, np.inf, np.inf, np.nan])
    assert_equal(a0, ex_a0)
    assert_equal(b0, ex_b0)
    assert a0.shape == ex_a0.shape
    assert b0.shape == ex_b0.shape

    a1, b1 = stats.norm.support([], [])
    ex_a1, ex_b1 = np.array([]), np.array([])
    assert_equal(a1, ex_a1)
    assert_equal(b1, ex_b1)
    assert a1.shape == ex_a1.shape
    assert b1.shape == ex_b1.shape
    # 使用正态分布对象 `stats.norm` 的 `support` 方法计算给定参数的支持范围 `a2` 和 `b2`
    a2, b2 = stats.norm.support([0, 0, 0, 1], [-1])
    
    # 创建一个长度为 4 的 numpy 数组，元素值为 NaN，用来作为预期值数组 `ex_a2`
    ex_a2 = np.array(4*[np.nan])
    
    # 创建一个长度为 4 的 numpy 数组，元素值为 NaN，用来作为预期值数组 `ex_b2`
    ex_b2 = np.array(4*[np.nan])
    
    # 使用 `assert_equal` 断言函数检查 `a2` 和 `ex_a2` 是否相等
    assert_equal(a2, ex_a2)
    
    # 使用 `assert_equal` 断言函数检查 `b2` 和 `ex_b2` 是否相等
    assert_equal(b2, ex_b2)
    
    # 使用 `assert` 断言语句检查 `a2` 的形状是否与 `ex_a2` 的形状相同
    assert a2.shape == ex_a2.shape
    
    # 使用 `assert` 断言语句检查 `b2` 的形状是否与 `ex_b2` 的形状相同
    assert b2.shape == ex_b2.shape
# 定义测试函数，用于验证修复 gh14953 中的问题
def test_stats_broadcasting_gh14953_regression():
    # 设置位置参数和尺度参数
    loc = [0., 0.]
    scale = [[1.], [2.], [3.]]
    # 断言计算得到的正态分布方差
    assert_equal(stats.norm.var(loc, scale), [[1., 1.], [4., 4.], [9., 9.]])
    # 测试一些边缘情况
    loc = np.empty((0, ))
    scale = np.empty((1, 0))
    # 断言正态分布方差的形状
    assert stats.norm.var(loc, scale).shape == (1, 0)


# 对余弦分布的 cdf、sf、ppf 和 isf 方法的几个值进行检查。期望值是使用 mpmath 计算得到的。
@pytest.mark.parametrize('x, expected',
                         [(-3.14159, 4.956444476505336e-19),
                          (3.14, 0.9999999998928399)])
def test_cosine_cdf_sf(x, expected):
    # 断言余弦分布的累积分布函数值接近期望值
    assert_allclose(stats.cosine.cdf(x), expected)
    # 断言余弦分布的生存函数值接近期望值
    assert_allclose(stats.cosine.sf(-x), expected)


@pytest.mark.parametrize('p, expected',
                         [(1e-6, -3.1080612413765905),
                          (1e-17, -3.141585429601399),
                          (0.975, 2.1447547020964923)])
def test_cosine_ppf_isf(p, expected):
    # 断言余弦分布的百分位点函数值接近期望值
    assert_allclose(stats.cosine.ppf(p), expected)
    # 断言余弦分布的逆生存函数值接近期望值
    assert_allclose(stats.cosine.isf(p), -expected)


def test_cosine_logpdf_endpoints():
    # 计算余弦分布在端点处的对数概率密度函数值
    logp = stats.cosine.logpdf([-np.pi, np.pi])
    # 参考值使用 mpmath 计算，假设 `np.cos(-1)` 比预期值高四个浮点数。参见 gh-18382。
    assert_array_less(logp, -37.18838327496655)


def test_distr_params_lists():
    # distribution objects 是在 test_discrete_basic 中添加的额外分布。
    # 所有其他分布都是字符串（名称），因此我们只选择它们来比较两个列表是否匹配。
    discrete_distnames = {name for name, _ in distdiscrete}
    invdiscrete_distnames = {name for name, _ in invdistdiscrete}
    # 断言离散分布名称列表与其逆的名称列表匹配
    assert discrete_distnames == invdiscrete_distnames

    cont_distnames = {name for name, _ in distcont}
    invcont_distnames = {name for name, _ in invdistcont}
    # 断言连续分布名称列表与其逆的名称列表匹配
    assert cont_distnames == invcont_distnames


def test_moment_order_4():
    # gh-13655 报告，如果一个分布具有接受 `moments` 参数的 `_stats` 方法，
    # 那么如果调用分布的 `moment` 方法时 `order=4`，则会调用更快/更精确的 `_stats` 方法，
    # 但结果不会被使用，仍会调用通用的 `_munp` 方法来计算矩。这个测试检查问题是否已经修复。
    # stats.skewnorm._stats 接受 `moments` 关键字
    stats.skewnorm._stats(a=0, moments='k')  # 没有失败 = 具有 `moments`
    # 当调用 `moment` 时，使用了 `_stats`，因此矩非常精确（等于正态分布的峰度，即3）
    assert stats.skewnorm.moment(order=4, a=0) == 3.0
    # 在 gh-13655 时，skewnorm._munp() 使用通用方法计算其结果，效率低且不太精确。
    # 当时下面的断言会失败。 skewnorm._munp()
    # 使用 assert 语句进行断言测试，验证 skewnorm 模块中的 _munp 函数
    # 在参数为 4 和 0 时的返回值是否等于 3.0
    assert stats.skewnorm._munp(4, 0) == 3.0
# 定义一个测试类 TestRelativisticBW，用于测试相对论布莱特-维格纳分布函数
class TestRelativisticBW:

    # 为 ROOT_pdf_sample_data 方法添加 pytest fixture，提供基于 CERN 的 ROOT 计算的样本数据点
    @pytest.fixture
    def ROOT_pdf_sample_data(self):
        """Sample data points for pdf computed with CERN's ROOT

        See - https://root.cern/

        Uses ROOT.TMath.BreitWignerRelativistic, available in ROOT
        versions 6.27+

        pdf calculated for Z0 Boson, W Boson, and Higgs Boson for
        x in `np.linspace(0, 200, 401)`.
        """
        # 加载相对路径下的样本数据文件，并转换成结构化数组
        data = np.load(
            Path(__file__).parent /
            'data/rel_breitwigner_pdf_sample_data_ROOT.npy'
        )
        data = np.rec.fromarrays(data.T, names='x,pdf,rho,gamma')
        return data

    # 使用 pytest.mark.parametrize 装饰器标记，对 test_pdf_against_ROOT 方法进行参数化测试
    @pytest.mark.parametrize(
        "rho,gamma,rtol", [
            (36.545206797050334, 2.4952, 5e-14),  # Z0 Boson
            (38.55107913669065, 2.085, 1e-14),   # W Boson
            (96292.3076923077, 0.0013, 5e-13),   # Higgs Boson
        ]
    )
    # 测试相对论布莱特-维格纳分布函数与 ROOT 实现的一致性
    def test_pdf_against_ROOT(self, ROOT_pdf_sample_data, rho, gamma, rtol):
        # 从样本数据中选择匹配 rho 和 gamma 的数据
        data = ROOT_pdf_sample_data[
            (ROOT_pdf_sample_data['rho'] == rho)
            & (ROOT_pdf_sample_data['gamma'] == gamma)
        ]
        x, pdf = data['x'], data['pdf']
        # 断言 ROOT 计算结果与 Python 实现的函数计算结果的近似程度
        assert_allclose(
            pdf, stats.rel_breitwigner.pdf(x, rho, scale=gamma), rtol=rtol
        )

    # 使用 pytest.mark.parametrize 装饰器标记，对 test_pdf_against_simple_implementation 方法进行参数化测试
    @pytest.mark.parametrize("rho, Gamma, rtol", [
        (36.545206797050334, 2.4952, 5e-13),   # Z0 Boson
        (38.55107913669065, 2.085, 5e-13),     # W Boson
        (96292.3076923077, 0.0013, 5e-10),     # Higgs Boson
    ])
    # 测试相对论布莱特-维格纳分布函数与基于维基百科公式的简单实现的一致性
    def test_pdf_against_simple_implementation(self, rho, Gamma, rtol):
        # 定义基于维基百科公式的概率密度函数实现
        def pdf(E, M, Gamma):
            gamma = np.sqrt(M**2 * (M**2 + Gamma**2))
            k = (2 * np.sqrt(2) * M * Gamma * gamma
                 / (np.pi * np.sqrt(M**2 + gamma)))
            return k / ((E**2 - M**2)**2 + M**2*Gamma**2)

        # 在一定范围内评估累积分布函数
        p = np.linspace(0.05, 0.95, 10)
        x = stats.rel_breitwigner.ppf(p, rho, scale=Gamma)
        # 计算 Python 实现和参考实现的结果
        res = stats.rel_breitwigner.pdf(x, rho, scale=Gamma)
        ref = pdf(x, rho*Gamma, Gamma)
        # 断言 Python 实现的结果与参考实现的结果的近似程度
        assert_allclose(res, ref, rtol=rtol)

    # 使用 pytest.mark.xslow 标记此测试方法为慢速测试，并对测试方法进行参数化
    @pytest.mark.parametrize(
        "rho,gamma", [
            pytest.param(
                36.545206797050334, 2.4952, marks=pytest.mark.slow
            ),   # Z0 Boson
            pytest.param(
                38.55107913669065, 2.085, marks=pytest.mark.xslow
            ),   # W Boson
            pytest.param(
                96292.3076923077, 0.0013, marks=pytest.mark.xslow
            ),   # Higgs Boson
        ]
    )
    # 定义一个测试方法，用于测试在 floc 设置时的拟合情况
    def test_fit_floc(self, rho, gamma):
        """Tests fit for cases where floc is set.

        `rel_breitwigner` has special handling for these cases.
        """
        # 设置一个种子以确保随机数生成的一致性
        seed = 6936804688480013683
        # 使用默认的随机数生成器创建一个实例
        rng = np.random.default_rng(seed)
        # 从相对 Breit-Wigner 分布中生成随机样本数据
        data = stats.rel_breitwigner.rvs(
            rho, scale=gamma, size=1000, random_state=rng
        )
        # 对生成的数据进行相对 Breit-Wigner 拟合，设置 floc=0
        fit = stats.rel_breitwigner.fit(data, floc=0)
        # 断言拟合结果的位置参数和尺度参数与预期值 rho 和 gamma 很接近
        assert_allclose((fit[0], fit[2]), (rho, gamma), rtol=2e-1)
        # 断言拟合结果的形状参数为 0
        assert fit[1] == 0
        # 再次进行拟合，这次设置 fscale=gamma
        fit = stats.rel_breitwigner.fit(data, floc=0, fscale=gamma)
        # 断言拟合结果的位置参数与预期值 rho 很接近
        assert_allclose(fit[0], rho, rtol=1e-2)
        # 断言拟合结果的形状参数为 0，尺度参数与预期值 gamma 很接近
        assert (fit[1], fit[2]) == (0, gamma)
class TestJohnsonSU:
    # 定义测试类 TestJohnsonSU
    @pytest.mark.parametrize("case", [  # a, b, loc, scale, m1, m2, g1, g2
            (-0.01, 1.1, 0.02, 0.0001, 0.02000137427557091,
             2.1112742956578063e-08, 0.05989781342460999, 20.36324408592951-3),
            (2.554395574161155, 2.2482281679651965, 0, 1, -1.54215386737391,
             0.7629882028469993, -1.256656139406788, 6.303058419339775-3)])
    # 参数化装饰器，为 test_moment_gh18071 提供两组参数化测试数据
    def test_moment_gh18071(self, case):
        # 测试函数 test_moment_gh18071，用例说明 gh-18071 报告了由 johnsonsu.stats 发出的 IntegrationWarning
        # 确保不再发出警告，并且结果与 Mathematica 的值精确匹配
        # 参考值来自 Mathematica，例如 Mean[JohnsonDistribution["SU",-0.01, 1.1, 0.02, 0.0001]]
        res = stats.johnsonsu.stats(*case[:4], moments='mvsk')
        # 使用 assert_allclose 检查 res 和 case[4:] 的近似程度，相对误差容差为 1e-14
        assert_allclose(res, case[4:], rtol=1e-14)


class TestTruncPareto:
    # 定义测试类 TestTruncPareto
    def test_pdf(self):
        # 测试 PDF 函数，使用截断 Pareto 分布的概率密度函数
        b, c = 1.8, 5.3
        # 生成线性空间上的均匀分布的数据点
        x = np.linspace(1.8, 5.3)
        # 计算截断 Pareto 分布在 x 上的概率密度函数值
        res = stats.truncpareto(b, c).pdf(x)
        # 计算参考值，为 Pareto 分布的概率密度函数除以累积分布函数
        ref = stats.pareto(b).pdf(x) / stats.pareto(b).cdf(c)
        # 使用 assert_allclose 检查 res 和 ref 的近似程度
        assert_allclose(res, ref)

    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    @pytest.mark.parametrize('fix_b', [True, False])
    @pytest.mark.parametrize('fix_c', [True, False])
    # 参数化装饰器，为 test_fit 提供多组参数化测试数据
    def test_fit(self, fix_loc, fix_scale, fix_b, fix_c):

        rng = np.random.default_rng(6747363148258237171)
        b, c, loc, scale = 1.8, 5.3, 1, 2.5
        # 创建截断 Pareto 分布对象 dist
        dist = stats.truncpareto(b, c, loc=loc, scale=scale)
        # 生成符合 dist 分布的随机数据
        data = dist.rvs(size=500, random_state=rng)

        kwds = {}
        # 根据参数 fix_loc, fix_scale, fix_b, fix_c 设置关键字参数 kwds
        if fix_loc:
            kwds['floc'] = loc
        if fix_scale:
            kwds['fscale'] = scale
        if fix_b:
            kwds['f0'] = b
        if fix_c:
            kwds['f1'] = c

        if fix_loc and fix_scale and fix_b and fix_c:
            # 如果所有参数都被固定，抛出运行时错误，消息指示已经没有需要优化的参数
            message = "All parameters fixed. There is nothing to optimize."
            with pytest.raises(RuntimeError, match=message):
                # 使用 pytest.raises 检查是否抛出预期的 RuntimeError 异常
                stats.truncpareto.fit(data, **kwds)
        else:
            # 否则调用 _assert_less_or_close_loglike 函数进行检查
            _assert_less_or_close_loglike(stats.truncpareto, data, **kwds)


class TestKappa3:
    # 定义测试类 TestKappa3
    def test_sf(self):
        # 测试 sf 函数，验证在 gh-18822 开发过程中发现的 kappa3.sf 是否会溢出
        # 检查最终实现中是否还会出现这种情况
        sf0 = 1 - stats.kappa3.cdf(0.5, 1e5)
        sf1 = stats.kappa3.sf(0.5, 1e5)
        # 使用 assert_allclose 检查 sf1 和 sf0 的近似程度
        assert_allclose(sf1, sf0)


class TestIrwinHall:
    # 定义测试类 TestIrwinHall
    unif = stats.uniform(0, 1)
    # 创建均匀分布对象 unif
    ih1 = stats.irwinhall(1)
    # 创建 Irwin-Hall 分布对象 ih1，参数为 1
    ih10 = stats.irwinhall(10)
    # 创建 Irwin-Hall 分布对象 ih10，参数为 10
    # 测试 IH(10) 分布的统计量（均值、方差、偏度、峰度）
    def test_stats_ih10(self):
        # 从 Wolfram Alpha 获取的 IH(10) 分布的均值、方差、偏度、峰度
        # Wolfram Alpha 使用 Pearson's 定义的峰度，因此需要减去 3
        # 应该是精确的整数除法转换为 fp64，不需要其他操作
        assert_array_max_ulp(self.ih10.stats('mvsk'), (5, 10/12, 0, -3/25))

    # 测试 IH(10) 分布的矩
    def test_moments_ih10(self):
        # 从 Wolfram Alpha 获取的 IH(10) 分布的矩值
        # 算法应该使用整数除法转换为 fp64，不需要其他操作
        # 因此这些值应该是精确的 ulpm（最小单位舍入误差），如果不是，也应该非常接近
        vals = [5, 155 / 6, 275 / 2, 752, 12650 / 3,
                677465 / 28, 567325 / 4,
                15266213 / 18, 10333565 / 2]
        moments = [self.ih10.moment(n+1) for n in range(len(vals))]
        assert_array_max_ulp(moments, vals)
        # 也来自 Wolfram Alpha 的 IH(10) 分布的第50阶矩
        m50 = self.ih10.moment(50)
        m50_exact = 17453002755350010529309685557285098151740985685/4862
        assert_array_max_ulp(m50, m50_exact)

    # 测试 IH(1) 分布的概率密度函数（PDF）
    def test_pdf_ih1_unif(self):
        # IH(1) 分布的 PDF 根据定义应该是 U(0,1)
        # 虽然浮点数计算顺序可能会有差异
        # 除非使用四倍精度，否则很难达到双精度的单个 ulp
        # 否则（在 sf/cdf/pdf 等方面）我们大约在6-10 ulp之内，这已经相当不错了
        pts = np.linspace(0, 1, 100)
        pdf_unif = self.unif.pdf(pts)
        pdf_ih1 = self.ih1.pdf(pts)
        assert_array_max_ulp(pdf_ih1, pdf_unif, maxulp=10)

    # 测试 IH(2) 分布的概率密度函数（PDF）
    def test_pdf_ih2_triangle(self):
        # IH(2) 分布的 PDF 是一个三角形
        ih2 = stats.irwinhall(2)
        npts = 101
        pts = np.linspace(0, 2, npts)
        expected = np.linspace(0, 2, npts)
        expected[(npts + 1) // 2:] = 2 - expected[(npts + 1) // 2:]
        pdf_ih2 = ih2.pdf(pts)
        assert_array_max_ulp(pdf_ih2, expected, maxulp=10)

    # 测试 IH(1) 分布的累积分布函数（CDF）
    def test_cdf_ih1_unif(self):
        # IH(1) 的 CDF 应该与均匀分布完全相同
        pts = np.linspace(0, 1, 100)
        cdf_unif = self.unif.cdf(pts)
        cdf_ih1 = self.ih1.cdf(pts)
        assert_array_max_ulp(cdf_ih1, cdf_unif, maxulp=10)

    # 测试 IH 分布的累积分布函数（CDF）
    def test_cdf(self):
        # IH 分布是对称的，因此在 n/2 处的 CDF 应该为 0.5
        n = np.arange(1, 10)
        ih = stats.irwinhall(n)
        ih_cdf = ih.cdf(n / 2)
        exact = np.repeat(1/2, len(n))
        # 应该完全等于 1/2，但浮点数计算顺序的差异可能会发生
        assert_array_max_ulp(ih_cdf, exact, maxulp=10)
    def test_cdf_ih10_exact(self):
        # 从 Wolfram Alpha 获取 "values CDF[UniformSumDistribution[10], x] x=0 to x=10"
        # 对称于 n/2，即 cdf[n-x] = 1-cdf[x] = sf[x]
        vals = [0, 1 / 3628800, 169 / 604800, 24427 / 1814400,
                  252023 / 1814400, 1 / 2, 1562377 / 1814400,
                  1789973 / 1814400, 604631 / 604800,
                  3628799 / 3628800, 1]

        # 主要是 bspline 评估的测试
        # 这个测试和其他测试主要是为了检测回归问题
        assert_array_max_ulp(self.ih10.cdf(np.arange(11)), vals, maxulp=10)

        assert_array_max_ulp(self.ih10.cdf(1/10), 1/36288000000000000, maxulp=10)
        ref = 36287999999999999/36288000000000000
        assert_array_max_ulp(self.ih10.cdf(99/10), ref, maxulp=10)

    def test_pdf_ih10_exact(self):
        # 从 Wolfram Alpha 获取 "values PDF[UniformSumDistribution[10], x] x=0 to x=10"
        # 对称于 n/2 = 5
        vals = [0, 1 / 362880, 251 / 181440, 913 / 22680, 44117 / 181440]
        vals += [15619 / 36288] + vals[::-1]
        assert_array_max_ulp(self.ih10.pdf(np.arange(11)), vals, maxulp=10)

    def test_sf_ih10_exact(self):
        assert_allclose(self.ih10.sf(np.arange(11)), 1 - self.ih10.cdf(np.arange(11)))
        # 从 Wolfram Alpha 获取 "SurvivalFunction[UniformSumDistribution[10],x] at x=1/10"
        # 对称于 n/2 = 5
        # 在 x=9.9 处，W|A 返回 1 作为 CDF
        ref = 36287999999999999/36288000000000000
        assert_array_max_ulp(self.ih10.sf(1/10), ref, maxulp=10)
# 参数化测试，用于多个分布名称和测试参数的组合
@pytest.mark.parametrize("case", [("kappa3", None, None, None, None),
                                  ("loglaplace", None, None, None, None),
                                  ("lognorm", None, None, None, None),
                                  ("lomax", None, None, None, None),
                                  ("pareto", None, None, None, None),])
def test_sf_isf_overrides(case):
    # 测试 SF（Survival Function）是否为 ISF（Inverse Survival Function）的逆操作。
    # 这个测试补充了对具有覆盖 sf 和 isf 方法的分布的基本检查。
    
    # 从参数元组中获取分布名称以及可能的特定参数
    distname, lp1, lp2, atol, rtol = case

    # 计算中位数处的概率质量的对数值（log10）
    lpm = np.log10(0.5)
    
    # 如果参数为 None，则使用默认值
    lp1 = lp1 or -290
    lp2 = lp2 or -14
    atol = atol or 0
    rtol = rtol or 1e-12
    
    # 获取与分布名称对应的分布函数对象
    dist = getattr(stats, distname)
    
    # 从预定义的参数字典中获取特定分布的参数
    params = dict(distcont)[distname]
    
    # 使用给定的参数初始化冻结分布对象
    dist_frozen = dist(*params)

    # 测试从非常深的右尾到中位数的部分。可以使用随机（对数均匀分布）点进行基准测试，但严格对数间隔的点对于测试是可以接受的。
    ref = np.logspace(lp1, lpm)
    res = dist_frozen.sf(dist_frozen.isf(ref))
    assert_allclose(res, ref, atol=atol, rtol=rtol)

    # 测试从中位数到左尾的部分
    ref = 1 - np.logspace(lp2, lpm, 20)
    res = dist_frozen.sf(dist_frozen.isf(ref))
    assert_allclose(res, ref, atol=atol, rtol=rtol)
```