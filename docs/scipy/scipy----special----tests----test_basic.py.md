# `D:\src\scipysrc\scipy\scipy\special\tests\test_basic.py`

```
# this program corresponds to special.py

### Means test is not done yet
# E   Means test is giving error (E)
# F   Means test is failing (F)
# EF  Means test is giving error and Failing
#!   Means test is segfaulting
# 8   Means test runs forever

###  test_besselpoly
###  test_mathieu_a
###  test_mathieu_even_coef
###  test_mathieu_odd_coef
###  test_modfresnelp
###  test_modfresnelm
#    test_pbdv_seq
###  test_pbvv_seq
###  test_sph_harm

import functools  # 导入functools模块，用于高阶函数的支持
import itertools  # 导入itertools模块，提供用于创建和操作迭代器的函数
import operator   # 导入operator模块，提供对Python内置操作符的函数形式的访问
import platform   # 导入platform模块，用于访问底层平台数据（如操作系统类型、版本等）
import sys        # 导入sys模块，提供了与Python解释器相关的功能

import numpy as np  # 导入NumPy库并重命名为np，用于支持多维数组与矩阵运算
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
        log, zeros, sqrt, asarray, inf, nan_to_num, real, arctan, double,
        array_equal)  # 从NumPy中导入多个函数和常量

import pytest  # 导入pytest库，用于编写和运行测试用例
from pytest import raises as assert_raises  # 导入pytest的raises方法并重命名为assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
        assert_array_equal, assert_array_almost_equal, assert_approx_equal,
        assert_, assert_allclose, assert_array_almost_equal_nulp,
        suppress_warnings)  # 导入NumPy测试模块中的多个断言函数

from scipy import special  # 导入SciPy库中的special子模块，提供特殊函数的实现
import scipy.special._ufuncs as cephes  # 导入SciPy库中特殊函数实现的C语言包装接口
from scipy.special import ellipe, ellipk, ellipkm1  # 导入Elliptic函数相关的特殊函数

from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj  # 导入更多Elliptic函数的实现
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2  # 导入Mathieu函数相关的特殊函数
from scipy._lib._util import np_long, np_ulong  # 导入SciPy库中的一些底层工具函数

from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
    _FACTORIALK_LIMITS_32BITS  # 导入特殊函数基础功能模块中的常量

from scipy.special._testutils import with_special_errors, \
     assert_func_equal, FuncData  # 导入特殊函数测试工具模块中的测试装饰器和函数

import math  # 导入Python标准库中的math模块，提供数学函数实现

class TestCephes:
    def test_airy(self):
        cephes.airy(0)  # 调用cephes模块中的airy函数进行测试

    def test_airye(self):
        cephes.airye(0)  # 调用cephes模块中的airye函数进行测试

    def test_binom(self):
        n = np.array([0.264, 4, 5.2, 17])  # 创建包含浮点数的NumPy数组n
        k = np.array([2, 0.4, 7, 3.3])     # 创建包含浮点数的NumPy数组k
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T  # 广播n和k数组，形成组合的可能性数组nk
        rknown = np.array([[-0.097152, 0.9263051596159367, 0.01858423645695389,
            -0.007581020651518199],[6, 2.0214389119675666, 0, 2.9827344527963846],
            [10.92, 2.22993515861399, -0.00585728, 10.468891352063146],
            [136, 3.5252179590758828, 19448, 1024.5526916174495]])  # 预期的结果数组rknown
        assert_func_equal(cephes.binom, rknown.ravel(), nk, rtol=1e-13)  # 使用自定义的测试函数assert_func_equal进行binom函数的测试

        # Test branches in implementation
        np.random.seed(1234)  # 设定随机数种子
        n = np.r_[np.arange(-7, 30), 1000*np.random.rand(30) - 500]  # 创建包含数组的NumPy数组n
        k = np.arange(0, 102)  # 创建从0到101的NumPy数组k
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T  # 广播n和k数组，形成组合的可能性数组nk

        assert_func_equal(cephes.binom,
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),
                          nk,
                          atol=1e-10, rtol=1e-10)  # 使用自定义的测试函数assert_func_equal进行binom函数的进一步测试
    def test_binom_2(self):
        # 测试二项分布函数的不同分支
        np.random.seed(1234)  # 设置随机种子为1234
        n = np.r_[np.logspace(1, 300, 20)]  # 生成1到300的对数间隔数组，与0到102的整数数组进行广播
        k = np.arange(0, 102)  # 生成0到101的整数数组
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T  # 将广播后的数组堆叠为二维数组，转置后reshape为一维数组

        assert_func_equal(cephes.binom,  # 调用断言函数，比较两个函数的输出结果是否相等
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),  # 调用cephes库中的二项分布函数，传入调整后的参数nk
                          nk,  # 传入参数nk
                          atol=1e-10, rtol=1e-10)  # 设置断言函数的容差值

    def test_binom_exact(self):
        @np.vectorize  # 声明一个numpy向量化函数
        def binom_int(n, k):
            n = int(n)  # 将n转换为整数
            k = int(k)  # 将k转换为整数
            num = 1  # 初始化num为1
            den = 1  # 初始化den为1
            for i in range(1, k+1):  # 循环计算组合数的分子和分母
                num *= i + n - k
                den *= i
            return float(num/den)  # 返回组合数的浮点数结果

        np.random.seed(1234)  # 设置随机种子为1234
        n = np.arange(1, 15)  # 生成1到14的整数数组
        k = np.arange(0, 15)  # 生成0到14的整数数组
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T  # 将广播后的数组堆叠为二维数组，转置后reshape为一维数组
        nk = nk[nk[:,0] >= nk[:,1]]  # 选择n >= k的数据点

        assert_func_equal(cephes.binom,  # 调用断言函数，比较两个函数的输出结果是否相等
                          binom_int(nk[:,0], nk[:,1]),  # 调用自定义的二项分布整数函数，传入参数nk
                          nk,  # 传入参数nk
                          atol=0, rtol=0)  # 设置断言函数的容差值为0

    def test_binom_nooverflow_8346(self):
        # 测试二项分布函数在不早期溢出的情况
        dataset = [
            (1000, 500, 2.70288240945436551e+299),
            (1002, 501, 1.08007396880791225e+300),
            (1004, 502, 4.31599279169058121e+300),
            (1006, 503, 1.72468101616263781e+301),
            (1008, 504, 6.89188009236419153e+301),
            (1010, 505, 2.75402257948335448e+302),
            (1012, 506, 1.10052048531923757e+303),
            (1014, 507, 4.39774063758732849e+303),
            (1016, 508, 1.75736486108312519e+304),
            (1018, 509, 7.02255427788423734e+304),
            (1020, 510, 2.80626776829962255e+305),
            (1022, 511, 1.12140876377061240e+306),
            (1024, 512, 4.48125455209897109e+306),
            (1026, 513, 1.79075474304149900e+307),
            (1028, 514, 7.15605105487789676e+307)
        ]
        dataset = np.asarray(dataset)  # 将数据集转换为numpy数组
        FuncData(cephes.binom, dataset, (0, 1), 2, rtol=1e-12).check()  # 创建函数数据对象并检查其结果

    def test_bdtr(self):
        assert_equal(cephes.bdtr(1,1,0.5),1.0)  # 调用cephes库中的bdtr函数，检查其结果是否等于1.0

    def test_bdtri(self):
        assert_equal(cephes.bdtri(1,3,0.5),0.5)  # 调用cephes库中的bdtri函数，检查其结果是否等于0.5

    def test_bdtrc(self):
        assert_equal(cephes.bdtrc(1,3,0.5),0.5)  # 调用cephes库中的bdtrc函数，检查其结果是否等于0.5

    def test_bdtrin(self):
        assert_equal(cephes.bdtrin(1,0,1),5.0)  # 调用cephes库中的bdtrin函数，检查其结果是否等于5.0

    def test_bdtrik(self):
        cephes.bdtrik(1,3,0.5)  # 调用cephes库中的bdtrik函数，无返回值，仅检查是否正常执行

    def test_bei(self):
        assert_equal(cephes.bei(0),0.0)  # 调用cephes库中的bei函数，检查其结果是否等于0.0

    def test_beip(self):
        assert_equal(cephes.beip(0),0.0)  # 调用cephes库中的beip函数，检查其结果是否等于0.0

    def test_ber(self):
        assert_equal(cephes.ber(0),1.0)  # 调用cephes库中的ber函数，检查其结果是否等于1.0

    def test_berp(self):
        assert_equal(cephes.berp(0),0.0)  # 调用cephes库中的berp函数，检查其结果是否等于0.0

    def test_besselpoly(self):
        assert_equal(cephes.besselpoly(0,0,0),1.0)  # 调用cephes库中的besselpoly函数，检查其结果是否等于1.0
    # 测试特殊函数 btdtr，在 SciPy 1.12.0 中已弃用，测试其返回值是否为 1.0
    def test_btdtr(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtr(1, 1, 1)
        assert_equal(y, 1.0)

    # 测试特殊函数 btdtri，在 SciPy 1.12.0 中已弃用，测试其返回值是否为 1.0
    def test_btdtri(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtri(1, 1, 1)
        assert_equal(y, 1.0)

    # 测试特殊函数 btdtria 的返回值是否为 5.0
    def test_btdtria(self):
        assert_equal(cephes.btdtria(1,1,1),5.0)

    # 测试特殊函数 btdtrib 的返回值是否为 5.0
    def test_btdtrib(self):
        assert_equal(cephes.btdtrib(1,1,1),5.0)

    # 测试特殊函数 cbrt 的返回值是否接近 1.0
    def test_cbrt(self):
        assert_approx_equal(cephes.cbrt(1),1.0)

    # 测试特殊函数 chdtr 的返回值是否为 0.0
    def test_chdtr(self):
        assert_equal(cephes.chdtr(1,0),0.0)

    # 测试特殊函数 chdtrc 的返回值是否为 1.0
    def test_chdtrc(self):
        assert_equal(cephes.chdtrc(1,0),1.0)

    # 测试特殊函数 chdtri 的返回值是否为 0.0
    def test_chdtri(self):
        assert_equal(cephes.chdtri(1,1),0.0)

    # 测试特殊函数 chdtriv 的返回值是否为 5.0
    def test_chdtriv(self):
        assert_equal(cephes.chdtriv(0,0),5.0)

    # 测试特殊函数 chndtr 的返回值是否为 0.0
    # 以下数组中的值是使用 Wolfram Alpha 计算得出的，作为参考值
    # 每行包含 (x, nu, lam, 期望值)
    values = np.array([
        [25.00, 20.0, 400, 4.1210655112396197139e-57],
        [25.00, 8.00, 250, 2.3988026526832425878e-29],
        [0.001, 8.00, 40., 5.3761806201366039084e-24],
        [0.010, 8.00, 40., 5.45396231055999457039e-20],
        [20.00, 2.00, 107, 1.39390743555819597802e-9],
        [22.50, 2.00, 107, 7.11803307138105870671e-9],
        [25.00, 2.00, 107, 3.11041244829864897313e-8],
        [3.000, 2.00, 1.0, 0.62064365321954362734],
        [350.0, 300., 10., 0.93880128006276407710],
        [100.0, 13.5, 10., 0.99999999650104210949],
        [700.0, 20.0, 400, 0.99999999925680650105],
        [150.0, 13.5, 10., 0.99999999999999983046],
        [160.0, 13.5, 10., 0.99999999999999999518],  # 1.0
    ])
    # 测试特殊函数 chndtr 的返回值是否与预期值接近，设置相对误差为 1e-12
    cdf = cephes.chndtr(values[:, 0], values[:, 1], values[:, 2])
    assert_allclose(cdf, values[:, 3], rtol=1e-12)

    # 测试特殊函数 chndtr 在无穷大参数下的返回值是否接近 2.0
    assert_almost_equal(cephes.chndtr(np.inf, np.inf, 0), 2.0)
    # 测试特殊函数 chndtr 在参数为无穷大的情况下的返回值是否为 0.0
    assert_almost_equal(cephes.chndtr(2, 1, np.inf), 0.0)
    # 测试特殊函数 chndtr 在参数中包含 NaN 时是否返回 NaN
    assert_(np.isnan(cephes.chndtr(np.nan, 1, 2)))
    assert_(np.isnan(cephes.chndtr(5, np.nan, 2)))
    assert_(np.isnan(cephes.chndtr(5, 1, np.nan)))

    # 测试特殊函数 chndtridf 的返回值是否为 5.0
    def test_chndtridf(self):
        assert_equal(cephes.chndtridf(0,0,1),5.0)

    # 测试特殊函数 chndtrinc 的返回值是否为 5.0
    def test_chndtrinc(self):
        assert_equal(cephes.chndtrinc(0,1,0),5.0)

    # 测试特殊函数 chndtrix 的返回值是否为 0.0
    def test_chndtrix(self):
        assert_equal(cephes.chndtrix(0,1,0),0.0)

    # 测试特殊函数 cosdg 的返回值是否为 1.0
    def test_cosdg(self):
        assert_equal(cephes.cosdg(0),1.0)

    # 测试特殊函数 cosm1 的返回值是否为 0.0
    def test_cosm1(self):
        assert_equal(cephes.cosm1(0),0.0)

    # 测试特殊函数 cotdg 在角度为 45 度时的返回值是否接近 1.0
    def test_cotdg(self):
        assert_almost_equal(cephes.cotdg(45),1.0)

    # 测试特殊函数 dawsn 的返回值是否为 0.0
    def test_dawsn(self):
        assert_equal(cephes.dawsn(0),0.0)
    # 测试特殊函数 dawsn 在参数为 1.23 时的返回值是否接近 0.50053727749081767
    assert_allclose(cephes.dawsn(1.23), 0.50053727749081767)
    def test_diric(self):
        # 测试在接近 2pi 的倍数时的行为。回归测试，解决 gh-4001 中描述的问题。
        n_odd = [1, 5, 25]  # 奇数值的列表
        x = np.array(2*np.pi + 5e-5).astype(np.float32)  # 创建一个接近 2pi 的值，转换为单精度浮点数
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=7)  # 断言 diric 函数的返回值接近 1.0，精确度为 7 位小数
        x = np.array(2*np.pi + 1e-9).astype(np.float64)  # 创建一个接近 2pi 的值，转换为双精度浮点数
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)  # 断言 diric 函数的返回值接近 1.0，精确度为 15 位小数
        x = np.array(2*np.pi + 1e-15).astype(np.float64)  # 创建一个接近 2pi 的值，转换为双精度浮点数
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)  # 断言 diric 函数的返回值接近 1.0，精确度为 15 位小数
        if hasattr(np, 'float128'):
            # 如果 numpy 支持 float128 类型
            x = np.array(2*np.pi + 1e-12).astype(np.float128)  # 创建一个接近 2pi 的值，转换为扩展精度浮点数
            assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=19)  # 断言 diric 函数的返回值接近 1.0，精确度为 19 位小数

        n_even = [2, 4, 24]  # 偶数值的列表
        x = np.array(2*np.pi + 1e-9).astype(np.float64)  # 创建一个接近 2pi 的值，转换为双精度浮点数
        assert_almost_equal(special.diric(x, n_even), -1.0, decimal=15)  # 断言 diric 函数的返回值接近 -1.0，精确度为 15 位小数

        # 在一些不接近 pi 的倍数的值上进行测试
        x = np.arange(0.2*np.pi, 1.0*np.pi, 0.2*np.pi)  # 创建一个 numpy 数组，包含指定范围内以 0.2pi 为步长的值
        octave_result = [0.872677996249965, 0.539344662916632,
                         0.127322003750035, -0.206011329583298]  # 预期的 octave 结果
        assert_almost_equal(special.diric(x, 3), octave_result, decimal=15)  # 断言 diric 函数的返回值接近 octave_result，精确度为 15 位小数
    # 测试复数参数下的 expm1 函数
    def test_expm1_complex(self):
        expm1 = cephes.expm1  # 导入并定义 expm1 函数的别名
        # 检查 expm1(0 + 0j) 的计算结果是否为 0 + 0j
        assert_equal(expm1(0 + 0j), 0 + 0j)
        # 检查 expm1(complex(np.inf, 0)) 的计算结果是否为 complex(np.inf, 0)
        assert_equal(expm1(complex(np.inf, 0)), complex(np.inf, 0))
        # 检查 expm1(complex(np.inf, 1)) 的计算结果是否为 complex(np.inf, np.inf)
        assert_equal(expm1(complex(np.inf, 1)), complex(np.inf, np.inf))
        # 检查 expm1(complex(np.inf, 2)) 的计算结果是否为 complex(-np.inf, np.inf)
        assert_equal(expm1(complex(np.inf, 2)), complex(-np.inf, np.inf))
        # 检查 expm1(complex(np.inf, 4)) 的计算结果是否为 complex(-np.inf, -np.inf)
        assert_equal(expm1(complex(np.inf, 4)), complex(-np.inf, -np.inf))
        # 检查 expm1(complex(np.inf, 5)) 的计算结果是否为 complex(np.inf, -np.inf)
        assert_equal(expm1(complex(np.inf, 5)), complex(np.inf, -np.inf))
        # 检查 expm1(complex(1, np.inf)) 的计算结果是否为 complex(np.nan, np.nan)
        assert_equal(expm1(complex(1, np.inf)), complex(np.nan, np.nan))
        # 检查 expm1(complex(0, np.inf)) 的计算结果是否为 complex(np.nan, np.nan)
        assert_equal(expm1(complex(0, np.inf)), complex(np.nan, np.nan))
        # 检查 expm1(complex(np.inf, np.inf)) 的计算结果是否为 complex(np.inf, np.nan)
        assert_equal(expm1(complex(np.inf, np.inf)), complex(np.inf, np.nan))
        # 检查 expm1(complex(-np.inf, np.inf)) 的计算结果是否为 complex(-1, 0)
        assert_equal(expm1(complex(-np.inf, np.inf)), complex(-1, 0))
        # 检查 expm1(complex(-np.inf, np.nan)) 的计算结果是否为 complex(-1, 0)
        assert_equal(expm1(complex(-np.inf, np.nan)), complex(-1, 0))
        # 检查 expm1(complex(np.inf, np.nan)) 的计算结果是否为 complex(np.inf, np.nan)
        assert_equal(expm1(complex(np.inf, np.nan)), complex(np.inf, np.nan))
        # 检查 expm1(complex(0, np.nan)) 的计算结果是否为 complex(np.nan, np.nan)
        assert_equal(expm1(complex(0, np.nan)), complex(np.nan, np.nan))
        # 检查 expm1(complex(1, np.nan)) 的计算结果是否为 complex(np.nan, np.nan)
        assert_equal(expm1(complex(1, np.nan)), complex(np.nan, np.nan))
        # 检查 expm1(complex(np.nan, 1)) 的计算结果是否为 complex(np.nan, np.nan)
        assert_equal(expm1(complex(np.nan, 1)), complex(np.nan, np.nan))
        # 检查 expm1(complex(np.nan, np.nan)) 的计算结果是否为 complex(np.nan, np.nan)

    @pytest.mark.xfail(reason='The real part of expm1(z) bad at these points')
    def test_expm1_complex_hard(self):
        # 这个函数的实部在 z.real = -log(cos(z.imag)) 时难以评估
        y = np.array([0.1, 0.2, 0.3, 5, 11, 20])
        x = -np.log(np.cos(y))
        z = x + 1j*y

        # 使用 mpmath.expm1 进行评估，设置 dps=1000
        expected = np.array([-5.5507901846769623e-17+0.10033467208545054j,
                              2.4289354732893695e-18+0.20271003550867248j,
                              4.5235500262585768e-17+0.30933624960962319j,
                              7.8234305217489006e-17-3.3805150062465863j,
                             -1.3685191953697676e-16-225.95084645419513j,
                              8.7175620481291045e-17+2.2371609442247422j])
        found = cephes.expm1(z)
        # 检查实部与期望值的相对误差在 20 以内
        assert_array_almost_equal_nulp(found.real, expected.real, 20)

    # 测试 fdtr 函数
    def test_fdtr(self):
        # 检查 fdtr(1, 1, 0) 的计算结果是否为 0.0
        assert_equal(cephes.fdtr(1, 1, 0), 0.0)
        # 使用 Wolfram Alpha 计算结果：CDF[FRatioDistribution[1e-6, 5], 10]
        # 检查 fdtr(1e-6, 5, 10) 的计算结果与 Wolfram Alpha 的结果是否在给定的相对误差范围内
        assert_allclose(cephes.fdtr(1e-6, 5, 10), 0.9999940790193488,
                        rtol=1e-12)

    # 测试 fdtrc 函数
    def test_fdtrc(self):
        # 检查 fdtrc(1, 1, 0) 的计算结果是否为 1.0
        assert_equal(cephes.fdtrc(1, 1, 0), 1.0)
        # 使用 Wolfram Alpha 计算结果：
        #   1 - CDF[FRatioDistribution[2, 1/10], 1e10]
        # 检查 fdtrc(2, 0.1, 1e10) 的计算结果与 Wolfram Alpha 的结果是否在给定的相对误差范围内
        assert_allclose(cephes.fdtrc(2, 0.1, 1e10), 0.27223784621293512,
                        rtol=1e-12)
    def test_fdtri(self):
        # 使用 cephes 库计算 F 分布的分位数，检验结果是否接近预期值
        assert_allclose(cephes.fdtri(1, 1, [0.499, 0.501]),
                        array([0.9937365, 1.00630298]), rtol=1e-6)
        # 从 Wolfram Alpha 取得的值，用来验证特定参数下 F 分布的累积分布函数的反函数
        p = 0.8756751669632105666874
        assert_allclose(cephes.fdtri(0.1, 1, p), 3, rtol=1e-12)

    @pytest.mark.xfail(reason='Returns nan on i686.')
    def test_fdtri_mysterious_failure(self):
        # 检验 cephes 库在特定条件下计算 F 分布分位数的失败情况（预期返回 nan）
        assert_allclose(cephes.fdtri(1, 1, 0.5), 1)

    def test_fdtridfd(self):
        # 使用 cephes 库计算 F 分布的非中心参数
        assert_equal(cephes.fdtridfd(1,0,0),5.0)

    def test_fresnel(self):
        # 使用 cephes 库计算 Fresnel 积分函数
        assert_equal(cephes.fresnel(0),(0.0,0.0))

    def test_gamma(self):
        # 使用 cephes 库计算 Gamma 函数
        assert_equal(cephes.gamma(5),24.0)

    def test_gammainccinv(self):
        # 使用 cephes 库计算 Incomplete Gamma 函数的反函数
        assert_equal(cephes.gammainccinv(5,1),0.0)

    def test_gammaln(self):
        # 使用 cephes 库计算 Gamma 函数的自然对数
        cephes.gammaln(10)

    def test_gammasgn(self):
        # 使用 cephes 库计算 Gamma 函数的符号
        vals = np.array([-4, -3.5, -2.3, 1, 4.2], np.float64)
        assert_array_equal(cephes.gammasgn(vals), np.sign(cephes.rgamma(vals)))

    def test_gdtr(self):
        # 使用 cephes 库计算 Gamma 分布的累积分布函数
        assert_equal(cephes.gdtr(1,1,0),0.0)

    def test_gdtr_inf(self):
        # 使用 cephes 库计算 Gamma 分布的累积分布函数，其中输入参数包含正无穷
        assert_equal(cephes.gdtr(1,1,np.inf),1.0)

    def test_gdtrc(self):
        # 使用 cephes 库计算 Gamma 分布的补充累积分布函数
        assert_equal(cephes.gdtrc(1,1,0),1.0)

    def test_gdtria(self):
        # 使用 cephes 库计算 Gamma 分布的反函数
        assert_equal(cephes.gdtria(0,1,1),0.0)

    def test_gdtrib(self):
        # 使用 cephes 库计算修正后的 Gamma 分布的反函数
        cephes.gdtrib(1,0,1)
        # assert_equal(cephes.gdtrib(1,0,1),5.0)

    def test_gdtrix(self):
        # 使用 cephes 库计算修正后的 Gamma 分布的反函数
        cephes.gdtrix(1,1,.1)

    def test_hankel1(self):
        # 使用 cephes 库计算第一类 Hankel 函数
        cephes.hankel1(1,1)

    def test_hankel1e(self):
        # 使用 cephes 库计算指数形式的第一类 Hankel 函数
        cephes.hankel1e(1,1)

    def test_hankel2(self):
        # 使用 cephes 库计算第二类 Hankel 函数
        cephes.hankel2(1,1)

    def test_hankel2e(self):
        # 使用 cephes 库计算指数形式的第二类 Hankel 函数
        cephes.hankel2e(1,1)

    def test_hyp1f1(self):
        # 使用 cephes 库计算超几何函数 1F1
        assert_approx_equal(cephes.hyp1f1(1,1,1), exp(1.0))
        assert_approx_equal(cephes.hyp1f1(3,4,-6), 0.026056422099537251095)
        cephes.hyp1f1(1,1,1)

    def test_hyp2f1(self):
        # 使用 cephes 库计算超几何函数 2F1
        assert_equal(cephes.hyp2f1(1,1,1,0),1.0)

    def test_i0(self):
        # 使用 cephes 库计算修正的 Bessel 函数 I0
        assert_equal(cephes.i0(0),1.0)

    def test_i0e(self):
        # 使用 cephes 库计算修正的指数形式 Bessel 函数 I0
        assert_equal(cephes.i0e(0),1.0)

    def test_i1(self):
        # 使用 cephes 库计算修正的 Bessel 函数 I1
        assert_equal(cephes.i1(0),0.0)

    def test_i1e(self):
        # 使用 cephes 库计算修正的指数形式 Bessel 函数 I1
        assert_equal(cephes.i1e(0),0.0)

    def test_it2i0k0(self):
        # 使用 cephes 库计算修正的 Bessel 函数的转换函数
        cephes.it2i0k0(1)

    def test_it2j0y0(self):
        # 使用 cephes 库计算修正的 Bessel 函数的转换函数
        cephes.it2j0y0(1)

    def test_it2struve0(self):
        # 使用 cephes 库计算修正的 Struve 函数
        cephes.it2struve0(1)

    def test_itairy(self):
        # 使用 cephes 库计算 Airy 函数
        cephes.itairy(1)

    def test_iti0k0(self):
        # 使用 cephes 库计算修正的 Bessel 函数的转换函数
        assert_equal(cephes.iti0k0(0),(0.0,0.0))

    def test_itj0y0(self):
        # 使用 cephes 库计算修正的 Bessel 函数的转换函数
        assert_equal(cephes.itj0y0(0),(0.0,0.0))

    def test_itmodstruve0(self):
        # 使用 cephes 库计算修正的 Modulus Struve 函数
        assert_equal(cephes.itmodstruve0(0),0.0)

    def test_itstruve0(self):
        # 使用 cephes 库计算修正的 Struve 函数
        assert_equal(cephes.itstruve0(0),0.0)

    def test_iv(self):
        # 使用 cephes 库计算修正的 Bessel 函数 IV
        assert_equal(cephes.iv(1,0),0.0)

    def test_ive(self):
        # 使用 cephes 库计算修正的指数形式 Bessel 函数 IV
        assert_equal(cephes.ive(1,0),0.0)
    # 定义测试方法 test_j0，测试 cephes.j0 函数返回值是否为 1.0
    def test_j0(self):
        assert_equal(cephes.j0(0), 1.0)

    # 定义测试方法 test_j1，测试 cephes.j1 函数返回值是否为 0.0
    def test_j1(self):
        assert_equal(cephes.j1(0), 0.0)

    # 定义测试方法 test_jn，测试 cephes.jn 函数返回值是否为 1.0
    def test_jn(self):
        assert_equal(cephes.jn(0, 0), 1.0)

    # 定义测试方法 test_jv，测试 cephes.jv 函数返回值是否为 1.0
    def test_jv(self):
        assert_equal(cephes.jv(0, 0), 1.0)

    # 定义测试方法 test_jve，测试 cephes.jve 函数返回值是否为 1.0
    def test_jve(self):
        assert_equal(cephes.jve(0, 0), 1.0)

    # 定义测试方法 test_k0，调用 cephes.k0 函数计算，无返回值的验证
    def test_k0(self):
        cephes.k0(2)

    # 定义测试方法 test_k0e，调用 cephes.k0e 函数计算，无返回值的验证
    def test_k0e(self):
        cephes.k0e(2)

    # 定义测试方法 test_k1，调用 cephes.k1 函数计算，无返回值的验证
    def test_k1(self):
        cephes.k1(2)

    # 定义测试方法 test_k1e，调用 cephes.k1e 函数计算，无返回值的验证
    def test_k1e(self):
        cephes.k1e(2)

    # 定义测试方法 test_kei，调用 cephes.kei 函数计算，无返回值的验证
    def test_kei(self):
        cephes.kei(2)

    # 定义测试方法 test_keip，测试 cephes.keip 函数返回值是否为 0.0
    def test_keip(self):
        assert_equal(cephes.keip(0), 0.0)

    # 定义测试方法 test_ker，调用 cephes.ker 函数计算，无返回值的验证
    def test_ker(self):
        cephes.ker(2)

    # 定义测试方法 test_kerp，调用 cephes.kerp 函数计算，无返回值的验证
    def test_kerp(self):
        cephes.kerp(2)

    # 定义测试方法 test_kelvin，调用 cephes.kelvin 函数计算，无返回值的验证
    def test_kelvin(self):
        cephes.kelvin(2)

    # 定义测试方法 test_kn，测试 cephes.kn 函数返回值是否为 1.0
    def test_kn(self):
        cephes.kn(1, 1)

    # 定义测试方法 test_kolmogi，测试 cephes.kolmogi 函数返回值和处理 NaN 值
    def test_kolmogi(self):
        assert_equal(cephes.kolmogi(1), 0.0)
        assert_(np.isnan(cephes.kolmogi(np.nan)))

    # 定义测试方法 test_kolmogorov，测试 cephes.kolmogorov 函数返回值是否为 1.0
    def test_kolmogorov(self):
        assert_equal(cephes.kolmogorov(0), 1.0)

    # 定义测试方法 test_kolmogp，测试 cephes._kolmogp 函数返回值是否为 -0.0
    def test_kolmogp(self):
        assert_equal(cephes._kolmogp(0), -0.0)

    # 定义测试方法 test_kolmogc，测试 cephes._kolmogc 函数返回值是否为 0.0
    def test_kolmogc(self):
        assert_equal(cephes._kolmogc(0), 0.0)

    # 定义测试方法 test_kolmogci，测试 cephes._kolmogci 函数返回值和处理 NaN 值
    def test_kolmogci(self):
        assert_equal(cephes._kolmogci(0), 0.0)
        assert_(np.isnan(cephes._kolmogci(np.nan)))

    # 定义测试方法 test_kv，调用 cephes.kv 函数计算，无返回值的验证
    def test_kv(self):
        cephes.kv(1, 1)

    # 定义测试方法 test_kve，调用 cephes.kve 函数计算，无返回值的验证
    def test_kve(self):
        cephes.kve(1, 1)

    # 定义测试方法 test_log1p，测试 cephes.log1p 函数不同输入情况下的返回值
    def test_log1p(self):
        log1p = cephes.log1p
        assert_equal(log1p(0), 0.0)
        assert_equal(log1p(-1), -np.inf)
        assert_equal(log1p(-2), np.nan)
        assert_equal(log1p(np.inf), np.inf)

    # 定义测试方法 test_log1p_complex，测试 cephes.log1p 函数在复数输入情况下的返回值
    def test_log1p_complex(self):
        log1p = cephes.log1p
        c = complex
        assert_equal(log1p(0 + 0j), 0 + 0j)
        assert_equal(log1p(c(-1, 0)), c(-np.inf, 0))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            # 验证复数输入情况下的近似相等性和 NaN 处理
            assert_allclose(log1p(c(1, np.inf)), c(np.inf, np.pi/2))
            assert_equal(log1p(c(1, np.nan)), c(np.nan, np.nan))
            assert_allclose(log1p(c(-np.inf, 1)), c(np.inf, np.pi))
            assert_equal(log1p(c(np.inf, 1)), c(np.inf, 0))
            assert_allclose(log1p(c(-np.inf, np.inf)), c(np.inf, 3*np.pi/4))
            assert_allclose(log1p(c(np.inf, np.inf)), c(np.inf, np.pi/4))
            assert_equal(log1p(c(np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(-np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, np.inf)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, 1)), c(np.nan, np.nan))
            assert_equal(log1p(c(np.nan, np.nan)), c(np.nan, np.nan))

    # 定义测试方法 test_lpmv，测试 cephes.lpmv 函数返回值是否为 1.0
    def test_lpmv(self):
        assert_equal(cephes.lpmv(0, 0, 1), 1.0)

    # 定义测试方法 test_mathieu_a，测试 cephes.mathieu_a 函数返回值是否为 1.0
    def test_mathieu_a(self):
        assert_equal(cephes.mathieu_a(1, 0), 1.0)

    # 定义测试方法 test_mathieu_b，测试 cephes.mathieu_b 函数返回值是否为 1.0
    def test_mathieu_b(self):
        assert_equal(cephes.mathieu_b(1, 0), 1.0)
    def test_mathieu_cem(self):
        # 断言调用 cephes.mathieu_cem 函数，检查返回结果是否等于 (1.0, 0.0)
        assert_equal(cephes.mathieu_cem(1, 0, 0), (1.0, 0.0))

        # 定义一个向量化函数 ce_smallq，用于计算 Mathieu 函数的小参数近似
        @np.vectorize
        def ce_smallq(m, q, z):
            # 将角度转换为弧度
            z *= np.pi / 180
            if m == 0:
                # 返回 m=0 时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return 2**(-0.5) * (1 - .5*q*cos(2*z))
            elif m == 1:
                # 返回 m=1 时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return cos(z) - q/8 * cos(3*z)
            elif m == 2:
                # 返回 m=2 时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return cos(2*z) - q*(cos(4*z)/12 - 1/4)
            else:
                # 返回其他 m 值时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return cos(m*z) - q*(cos((m+2)*z)/(4*(m+1)) - cos((m-2)*z)/(4*(m-1)))
        
        # 创建 m 和 q 的数组，并对 cephes.mathieu_cem 的返回值和 ce_smallq 的返回值进行比较
        m = np.arange(0, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_cem(m[:, None], q[None, :], 0.123)[0],
                        ce_smallq(m[:, None], q[None, :], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_sem(self):
        # 断言调用 cephes.mathieu_sem 函数，检查返回结果是否等于 (0.0, 1.0)
        assert_equal(cephes.mathieu_sem(1, 0, 0), (0.0, 1.0))

        # 定义一个向量化函数 se_smallq，用于计算 Mathieu 函数的小参数近似
        @np.vectorize
        def se_smallq(m, q, z):
            # 将角度转换为弧度
            z *= np.pi / 180
            if m == 1:
                # 返回 m=1 时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return sin(z) - q/8 * sin(3*z)
            elif m == 2:
                # 返回 m=2 时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return sin(2*z) - q*sin(4*z)/12
            else:
                # 返回其他 m 值时的 Mathieu 函数的近似表达式，加上 O(q^2) 的修正
                return sin(m*z) - q*(sin((m+2)*z)/(4*(m+1)) - sin((m-2)*z)/(4*(m-1)))
        
        # 创建 m 和 q 的数组，并对 cephes.mathieu_sem 的返回值和 se_smallq 的返回值进行比较
        m = np.arange(1, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_sem(m[:, None], q[None, :], 0.123)[0],
                        se_smallq(m[:, None], q[None, :], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_modcem1(self):
        # 断言调用 cephes.mathieu_modcem1 函数，检查返回结果是否等于 (0.0, 0.0)
        assert_equal(cephes.mathieu_modcem1(1, 0, 0), (0.0, 0.0))

    def test_mathieu_modcem2(self):
        # 调用 cephes.mathieu_modcem2 函数，但不检查返回值

        # 测试反射关系 AMS 20.6.19
        m = np.arange(0, 4)[:, None, None]
        q = np.r_[np.logspace(-2, 2, 10)][None, :, None]
        z = np.linspace(0, 1, 7)[None, None, :]

        # 计算 cephes.mathieu_modcem2 在 z=-z 处的值
        y1 = cephes.mathieu_modcem2(m, q, -z)[0]

        # 计算反射系数 fr，并使用其计算 y2
        fr = -cephes.mathieu_modcem2(m, q, 0)[0] / cephes.mathieu_modcem1(m, q, 0)[0]
        y2 = (-cephes.mathieu_modcem2(m, q, z)[0]
              - 2 * fr * cephes.mathieu_modcem1(m, q, z)[0])

        # 断言 y1 和 y2 的近似相等性
        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_modsem1(self):
        # 断言调用 cephes.mathieu_modsem1 函数，检查返回结果是否等于 (0.0, 0.0)
        assert_equal(cephes.mathieu_modsem1(1, 0, 0), (0.0, 0.0))

    def test_mathieu_modsem2(self):
        # 调用 cephes.mathieu_modsem2 函数，但不检查返回值

        # 测试反射关系 AMS 20.6.20
        m = np.arange(1, 4)[:, None, None]
        q = np.r_[np.logspace(-2, 2, 10)][None, :, None]
        z = np.linspace(0, 1, 7)[None, None, :]

        # 计算 cephes.mathieu_modsem2 在 z=-z 处的值
        y1 = cephes.mathieu_modsem2(m, q, -z)[0]

        # 计算反射系数 fr，并使用其计算 y2
        fr = cephes.mathieu_modsem2(m, q, 0)[1] / cephes.mathieu_modsem1(m, q, 0)[1]
        y2 = (cephes.mathieu_modsem2(m, q, z)[0]
              - 2 * fr * cephes.mathieu_modsem1(m, q, z)[0])

        # 断言 y1 和 y2 的近似相等性
        assert_allclose(y1, y2, rtol=1e-10)
    def test_mathieu_overflow(self):
        # 检查这些函数调用是否返回 NaN 而不是导致 SEGV（段错误）
        assert_equal(cephes.mathieu_cem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_cem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem2(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem2(10000, 1.5, 1.3), (np.nan, np.nan))

    def test_mathieu_ticket_1847(self):
        # 回归测试 --- 这个调用可能会导致偶尔返回 NaN，曾经存在越界访问问题
        for k in range(60):
            v = cephes.mathieu_modsem2(2, 100, -1)
            # ACM TOMS 804 提供的值（通过数值微分导出）
            assert_allclose(v[0], 0.1431742913063671074347, rtol=1e-10)
            assert_allclose(v[1], 0.9017807375832909144719, rtol=1e-4)

    def test_modfresnelm(self):
        # 调用 cephes.modfresnelm 进行计算
        cephes.modfresnelm(0)

    def test_modfresnelp(self):
        # 调用 cephes.modfresnelp 进行计算
        cephes.modfresnelp(0)

    def test_modstruve(self):
        # 检查 cephes.modstruve(1,0) 的返回值是否等于 0.0
        assert_equal(cephes.modstruve(1,0),0.0)

    def test_nbdtr(self):
        # 检查 cephes.nbdtr(1,1,1) 的返回值是否等于 1.0
        assert_equal(cephes.nbdtr(1,1,1),1.0)

    def test_nbdtrc(self):
        # 检查 cephes.nbdtrc(1,1,1) 的返回值是否等于 0.0
        assert_equal(cephes.nbdtrc(1,1,1),0.0)

    def test_nbdtri(self):
        # 检查 cephes.nbdtri(1,1,1) 的返回值是否等于 1.0
        assert_equal(cephes.nbdtri(1,1,1),1.0)

    def test_nbdtrik(self):
        # 调用 cephes.nbdtrik(1,.4,.5) 进行计算
        cephes.nbdtrik(1,.4,.5)

    def test_nbdtrin(self):
        # 检查 cephes.nbdtrin(1,0,0) 的返回值是否等于 5.0
        assert_equal(cephes.nbdtrin(1,0,0),5.0)

    def test_ncfdtr(self):
        # 检查 cephes.ncfdtr(1,1,1,0) 的返回值是否等于 0.0
        assert_equal(cephes.ncfdtr(1,1,1,0),0.0)

    def test_ncfdtri(self):
        # 检查 cephes.ncfdtri(1, 1, 1, 0) 的返回值是否等于 0.0
        assert_equal(cephes.ncfdtri(1, 1, 1, 0), 0.0)
        f = [0.5, 1, 1.5]
        p = cephes.ncfdtr(2, 3, 1.5, f)
        # 检查 cephes.ncfdtri(2, 3, 1.5, p) 的返回值是否接近于 f 中的值
        assert_allclose(cephes.ncfdtri(2, 3, 1.5, p), f)

    def test_ncfdtridfd(self):
        dfd = [1, 2, 3]
        p = cephes.ncfdtr(2, dfd, 0.25, 15)
        # 检查 cephes.ncfdtridfd(2, p, 0.25, 15) 的返回值是否接近于 dfd
        assert_allclose(cephes.ncfdtridfd(2, p, 0.25, 15), dfd)

    def test_ncfdtridfn(self):
        dfn = [0.1, 1, 2, 3, 1e4]
        p = cephes.ncfdtr(dfn, 2, 0.25, 15)
        # 检查 cephes.ncfdtridfn(p, 2, 0.25, 15) 的返回值是否接近于 dfn
        assert_allclose(cephes.ncfdtridfn(p, 2, 0.25, 15), dfn, rtol=1e-5)

    def test_ncfdtrinc(self):
        nc = [0.5, 1.5, 2.0]
        p = cephes.ncfdtr(2, 3, nc, 15)
        # 检查 cephes.ncfdtrinc(2, 3, p, 15) 的返回值是否接近于 nc
        assert_allclose(cephes.ncfdtrinc(2, 3, p, 15), nc)
    # 测试 cephes.nctdtr 函数的不同参数组合是否返回预期的结果
    def test_nctdtr(self):
        # 断言：当 df = 1, t = 0, nc = 0 时，返回值应为 0.5
        assert_equal(cephes.nctdtr(1, 0, 0), 0.5)
        # 断言：当 df = 9, t = 65536, nc = 45 时，返回值应为 0.0
        assert_equal(cephes.nctdtr(9, 65536, 45), 0.0)

        # 断言：当 df -> ∞, t = 1.0, nc = 1.0 时，返回值应接近 0.5，精确到小数点后五位
        assert_approx_equal(cephes.nctdtr(np.inf, 1., 1.), 0.5, 5)
        # 断言：当 df = 2.0, t -> ∞, nc = 10.0 时，返回值应为 NaN
        assert_(np.isnan(cephes.nctdtr(2., np.inf, 10.)))
        # 断言：当 df = 2.0, t = 1.0, nc -> ∞ 时，返回值应为 1.0
        assert_approx_equal(cephes.nctdtr(2., 1., np.inf), 1.0)

        # 断言：当 df = NaN, t = 1.0, nc = 1.0 时，返回值应为 NaN
        assert_(np.isnan(cephes.nctdtr(np.nan, 1., 1.)))
        # 断言：当 df = 2.0, t = NaN, nc = 1.0 时，返回值应为 NaN
        assert_(np.isnan(cephes.nctdtr(2., np.nan, 1.)))
        # 断言：当 df = 2.0, t = 1.0, nc = NaN 时，返回值应为 NaN
        assert_(np.isnan(cephes.nctdtr(2., 1., np.nan)))

    # 测试 cephes.nctdtridf 函数的使用，输入参数为 (df, t, nc)
    def test_nctdtridf(self):
        cephes.nctdtridf(1, 0.5, 0)

    # 测试 cephes.nctdtrinc 函数的使用，输入参数为 (df, t, nc)
    def test_nctdtrinc(self):
        cephes.nctdtrinc(1, 0, 0)

    # 测试 cephes.nctdtrit 函数的使用，输入参数为 (p, q, x)
    def test_nctdtrit(self):
        cephes.nctdtrit(.1, 0.2, .5)

    # 测试 cephes.nrdtrimn 函数的使用，输入参数为 (p, q, x)
    def test_nrdtrimn(self):
        # 断言：当 p = 0.5, q = 1, x = 1 时，返回值应接近 1.0
        assert_approx_equal(cephes.nrdtrimn(0.5, 1, 1), 1.0)

    # 测试 cephes.nrdtrisd 函数的使用，输入参数为 (p, q, x)
    def test_nrdtrisd(self):
        # 断言：当 p = 0.5, q = 0.5, x = 0.5 时，返回值应接近 0.0，允许绝对误差为 0，相对误差为 0
        assert_allclose(cephes.nrdtrisd(0.5, 0.5, 0.5), 0.0, atol=0, rtol=0)

    # 测试 cephes.obl_ang1 函数的使用，输入参数为 (m, n, a, b)
    def test_obl_ang1(self):
        cephes.obl_ang1(1, 1, 1, 0)

    # 测试 cephes.obl_ang1_cv 函数的使用，输入参数为 (m, n, a, b, c)
    def test_obl_ang1_cv(self):
        # 断言：返回值的第一个元素应接近 1.0，第二个元素应接近 0.0
        result = cephes.obl_ang1_cv(1, 1, 1, 1, 0)
        assert_almost_equal(result[0], 1.0)
        assert_almost_equal(result[1], 0.0)

    # 测试 cephes.obl_cv 函数的使用，输入参数为 (m, n, a)
    def test_obl_cv(self):
        # 断言：返回值应为 2.0
        assert_equal(cephes.obl_cv(1, 1, 0), 2.0)

    # 测试 cephes.obl_rad1 函数的使用，输入参数为 (m, n, a, b)
    def test_obl_rad1(self):
        cephes.obl_rad1(1, 1, 1, 0.1)

    # 测试 cephes.obl_rad1_cv 函数的使用，输入参数为 (m, n, a, b, c)
    def test_obl_rad1_cv(self):
        cephes.obl_rad1_cv(1, 1, 1, 1, 0)

    # 测试 cephes.obl_rad2 函数的使用，输入参数为 (m, n, a, b)
    def test_obl_rad2(self):
        cephes.obl_rad2(1, 1, 1, 0)

    # 测试 cephes.obl_rad2_cv 函数的使用，输入参数为 (m, n, a, b, c)
    def test_obl_rad2_cv(self):
        cephes.obl_rad2_cv(1, 1, 1, 1, 0)

    # 测试 cephes.pbdv 函数的使用，输入参数为 (n, x)
    def test_pbdv(self):
        # 断言：当 n = 1, x = 0 时，返回值应为 (0.0, 1.0)
        assert_equal(cephes.pbdv(1, 0), (0.0, 1.0))

    # 测试 cephes.pbvv 函数的使用，输入参数为 (n, x)
    def test_pbvv(self):
        cephes.pbvv(1, 0)

    # 测试 cephes.pbwa 函数的使用，输入参数为 (n, x)
    def test_pbwa(self):
        cephes.pbwa(1, 0)

    # 测试 cephes.pdtr 函数的使用，输入参数为 (m, x)
    def test_pdtr(self):
        # 断言：当 m = 0, x = 1 时，返回值应接近 e^-1
        val = cephes.pdtr(0, 1)
        assert_almost_equal(val, np.exp(-1))
        # 边界情况：当 m 为数组 [0, 1, 2], x = 0 时，返回值应为 [1, 1, 1]
        val = cephes.pdtr([0, 1, 2], 0)
        assert_array_equal(val, [1, 1, 1])

    # 测试 cephes.pdtrc 函数的使用，输入参数为 (m, x)
    def test_pdtrc(self):
        # 断言：当 m = 0, x = 1 时，返回值应接近 1 - e^-1
        val = cephes.pdtrc(0, 1)
        assert_almost_equal(val, 1 - np.exp(-1))
        # 边界情况：当 m 为数组 [0, 1, 2], x = 0.0 时，返回值应为 [0, 0, 0]
        val = cephes.pdtrc([0, 1, 2], 0.0)
        assert_array_equal(val, [0, 0, 0])

    # 测试 cephes.pdtri 函数的使用，输入参数为 (p, q)
    def test_pdtri(self):
        # 使用 suppress_warnings 上下文管理器，过滤 RuntimeWarning: "floating point number truncated to an integer"
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "floating point number truncated to an integer")
            cephes.pdtri(0.5, 0.5)

    # 测试 cephes.pdtrik 函数的使用，输入参数为 (p, q)
    def test_pdtrik(self):
        # 断言：对于返回的 k，使用 cephes.gammaincc(k + 1,
    # 调用 cephes 模块中的 pro_rad2 函数，传入参数 (1,1,1,0)，并执行测试
    def test_pro_rad2(self):
        cephes.pro_rad2(1,1,1,0)

    # 调用 cephes 模块中的 pro_rad2_cv 函数，传入参数 (1,1,1,1,0)，并执行测试
    def test_pro_rad2_cv(self):
        cephes.pro_rad2_cv(1,1,1,1,0)

    # 调用 cephes 模块中的 psi 函数，传入参数 1，并执行测试
    def test_psi(self):
        cephes.psi(1)

    # 调用 cephes 模块中的 radian 函数，传入参数 (0,0,0)，并使用 assert_equal 进行断言测试结果是否为 0
    def test_radian(self):
        assert_equal(cephes.radian(0,0,0),0)

    # 调用 cephes 模块中的 rgamma 函数，传入参数 1，并使用 assert_equal 进行断言测试结果是否为 1.0
    def test_rgamma(self):
        assert_equal(cephes.rgamma(1),1.0)

    # 调用 cephes 模块中的 round 函数，传入不同的参数进行测试，使用 assert_equal 进行断言测试结果是否符合预期
    def test_round(self):
        assert_equal(cephes.round(3.4),3.0)
        assert_equal(cephes.round(-3.4),-3.0)
        assert_equal(cephes.round(3.6),4.0)
        assert_equal(cephes.round(-3.6),-4.0)
        assert_equal(cephes.round(3.5),4.0)
        assert_equal(cephes.round(-3.5),-4.0)

    # 调用 cephes 模块中的 shichi 函数，传入参数 1，并执行测试
    def test_shichi(self):
        cephes.shichi(1)

    # 调用 cephes 模块中的 sici 函数，传入参数 1，并执行测试；接着使用 assert_almost_equal 和 assert_ 进行更严格的断言测试
    def test_sici(self):
        cephes.sici(1)

        # 测试特定情况下 sici 返回值与预期 np.pi * 0.5 和 0 是否接近
        s, c = cephes.sici(np.inf)
        assert_almost_equal(s, np.pi * 0.5)
        assert_almost_equal(c, 0)

        # 测试特定情况下 sici 返回值与预期 -np.pi * 0.5 和 NaN 是否接近
        s, c = cephes.sici(-np.inf)
        assert_almost_equal(s, -np.pi * 0.5)
        assert_(np.isnan(c), "cosine integral(-inf) is not nan")

    # 调用 cephes 模块中的 sindg 函数，传入参数 90，并使用 assert_equal 进行断言测试结果是否为 1.0
    def test_sindg(self):
        assert_equal(cephes.sindg(90),1.0)

    # 调用 cephes 模块中的 smirnov 函数，传入参数 (1, .1)，并使用 assert_equal 进行断言测试结果是否为 0.9
    def test_smirnov(self):
        assert_equal(cephes.smirnov(1,.1),0.9)
        # 测试 NaN 输入时 smirnov 返回值是否为 NaN
        assert_(np.isnan(cephes.smirnov(1,np.nan)))

    # 调用 cephes 模块中的 _smirnovp 函数，传入参数 (1, .1)，并使用 assert_equal 进行断言测试结果是否符合预期
    def test_smirnovp(self):
        assert_equal(cephes._smirnovp(1, .1), -1)
        assert_equal(cephes._smirnovp(2, 0.75), -2*(0.25)**(2-1))
        assert_equal(cephes._smirnovp(3, 0.75), -3*(0.25)**(3-1))
        # 测试 NaN 输入时 _smirnovp 返回值是否为 NaN
        assert_(np.isnan(cephes._smirnovp(1, np.nan)))

    # 调用 cephes 模块中的 _smirnovc 函数，传入参数 (1, .1)，并使用 assert_equal 进行断言测试结果是否为 0.1
    def test_smirnovc(self):
        assert_equal(cephes._smirnovc(1,.1),0.1)
        # 测试 NaN 输入时 _smirnovc 返回值是否为 NaN
        assert_(np.isnan(cephes._smirnovc(1,np.nan)))
        # 使用 linspace 生成多个点，测试 smirnovc 返回值是否正确计算
        x10 = np.linspace(0, 1, 11, endpoint=True)
        assert_almost_equal(cephes._smirnovc(3, x10), 1-cephes.smirnov(3, x10))
        x4 = np.linspace(0, 1, 5, endpoint=True)
        assert_almost_equal(cephes._smirnovc(4, x4), 1-cephes.smirnov(4, x4))

    # 调用 cephes 模块中的 smirnovi 函数，传入参数 (1,0.4)，并使用 assert_almost_equal 进行断言测试结果是否接近 0.4
    def test_smirnovi(self):
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.4)),0.4)
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.6)),0.6)
        # 测试 NaN 输入时 smirnovi 返回值是否为 NaN
        assert_(np.isnan(cephes.smirnovi(1,np.nan)))

    # 调用 cephes 模块中的 _smirnovc 函数，传入参数 (1,0.4)，并使用 assert_almost_equal 进行断言测试结果是否接近 0.4
    def test_smirnovci(self):
        assert_almost_equal(cephes._smirnovc(1,cephes._smirnovci(1,0.4)),0.4)
        assert_almost_equal(cephes._smirnovc(1,cephes._smirnovci(1,0.6)),0.6)
        # 测试 NaN 输入时 smirnovci 返回值是否为 NaN
        assert_(np.isnan(cephes._smirnovci(1,np.nan)))

    # 调用 cephes 模块中的 spence 函数，传入参数 1，并使用 assert_equal 进行断言测试结果是否为 0.0
    def test_spence(self):
        assert_equal(cephes.spence(1),0.0)

    # 调用 cephes 模块中的 stdtr 函数，传入参数 (1,0)，并使用 assert_equal 进行断言测试结果是否为 0.5
    def test_stdtr(self):
        assert_equal(cephes.stdtr(1,0),0.5)
        assert_almost_equal(cephes.stdtr(1,1), 0.75)
        assert_almost_equal(cephes.stdtr(1,2), 0.852416382349)

    # 调用 cephes 模块中的 stdtridf 函数，传入参数 (0.7,1)，并执行测试
    def test_stdtridf(self):
        cephes.stdtridf(0.7,1)

    # 调用 cephes 模块中的 stdtrit 函数，传入参数 (1,0.7)，并执行测试
    def test_stdtrit(self):
        cephes.stdtrit(1,0.7)

    # 调用 cephes 模块中的 struve 函数，传入参数 (0,0)，并使用 assert_equal 进行断言测试结果是否为 0.0
    def test_struve(self):
        assert_equal(cephes.struve(0,0),0.0)

    # 调用 cephes 模块中的 tandg 函数，传入参数 45，并使用 assert_equal 进行断言测试结果是否为 1.0
    def test_tandg(self):
        assert_equal(cephes.tandg(45),1.0)

    # 调用 cephes 模块中的 tklmbda 函数，传入参数 (1,1)，
    # 调用 cephes 模块中的 y1 函数，计算第一个数值参数的贝塞尔函数 Y_1(x)
    def test_y1(self):
        cephes.y1(1)
    
    # 调用 cephes 模块中的 yn 函数，计算第一个和第二个数值参数的贝塞尔函数 Y_n(x)
    def test_yn(self):
        cephes.yn(1, 1)
    
    # 调用 cephes 模块中的 yv 函数，计算第一个和第二个数值参数的 modified Bessel 函数 Y_v(x)
    def test_yv(self):
        cephes.yv(1, 1)
    
    # 调用 cephes 模块中的 yve 函数，计算第一个和第二个数值参数的 exponentially scaled modified Bessel 函数 Y_v(x)
    def test_yve(self):
        cephes.yve(1, 1)
    # 定义一个名为 test_wofz 的测试方法，用于测试 cephes.wofz 函数的正确性
    def test_wofz(self):
        # 定义一个复数列表 z，包含多个复数值作为输入参数
        z = [complex(624.2,-0.26123), complex(-0.4,3.), complex(0.6,2.),
             complex(-1.,1.), complex(-1.,-9.), complex(-1.,9.),
             complex(-0.0000000234545,1.1234), complex(-3.,5.1),
             complex(-53,30.1), complex(0.0,0.12345),
             complex(11,1), complex(-22,-2), complex(9,-28),
             complex(21,-33), complex(1e5,1e5), complex(1e14,1e14)
             ]
        # 定义一个复数列表 w，包含多个复数值作为预期输出结果
        w = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                    0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                    -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                    0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                    -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                    8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                    -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                    -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                    -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                    -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                    0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                    0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                    -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                    3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                    -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                    2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                    2.82094791773878143474039725773333923127678361e-15)
        ]
        # 断言函数 assert_func_equal 调用 cephes.wofz 函数，比较其输出 w 是否等于预期输出 w，设置相对容差为 1e-13
        assert_func_equal(cephes.wofz, w, z, rtol=1e-13)
class TestAiry:
    def test_airy(self):
        # 测试 airy 函数，确保计算精度达到8位小数

        # 计算 special 模块中 airy 函数在 x = 0.99 处的返回值
        x = special.airy(.99)
        # 断言计算结果与预期值的近似相等，精度为8位小数
        assert_array_almost_equal(
            x,
            array([0.13689066, -0.16050153, 1.19815925, 0.92046818]),
            8,
        )

        # 计算 special 模块中 airy 函数在 x = 0.41 处的返回值
        x = special.airy(.41)
        # 断言计算结果与预期值的近似相等，精度为8位小数
        assert_array_almost_equal(
            x,
            array([0.25238916, -0.23480512, 0.80686202, 0.51053919]),
            8,
        )

        # 计算 special 模块中 airy 函数在 x = -0.36 处的返回值
        x = special.airy(-.36)
        # 断言计算结果与预期值的近似相等，精度为8位小数
        assert_array_almost_equal(
            x,
            array([0.44508477, -0.23186773, 0.44939534, 0.48105354]),
            8,
        )

    def test_airye(self):
        # 计算 special 模块中 airye 函数在 a = 0.01 处的返回值，并验证其精度达到6位小数

        # 计算 special 模块中 airy 函数在 b = 0.01 处的返回值
        b = special.airy(0.01)
        # 创建一个长度为4的空列表 b1
        b1 = [None] * 4

        # 计算 b1 中前两个元素
        for n in range(2):
            b1[n] = b[n] * exp(2.0 / 3.0 * 0.01 * sqrt(0.01))

        # 计算 b1 中后两个元素
        for n in range(2, 4):
            b1[n] = b[n] * exp(-abs(real(2.0 / 3.0 * 0.01 * sqrt(0.01))))

        # 断言计算结果 a 与预期值 b1 的近似相等，精度为6位小数
        assert_array_almost_equal(a, b1, 6)

    def test_bi_zeros(self):
        # 计算 special 模块中 bi_zeros 函数返回的结果，验证精度达到4位小数

        # 获取 bi_zeros 函数返回的结果 bi
        bi = special.bi_zeros(2)
        # 预期的返回结果 bia，包含四个数组的元组
        bia = (
            array([-1.17371322, -3.2710930]),
            array([-2.29443968, -4.07315509]),
            array([-0.45494438, 0.39652284]),
            array([0.60195789, -0.76031014]),
        )

        # 断言计算结果 bi 与预期值 bia 的近似相等，精度为4位小数
        assert_array_almost_equal(bi, bia, 4)

        # 再次调用 bi_zeros 函数，获取更多的结果
        bi = special.bi_zeros(5)
        # 断言 bi 的第一个元素与预期数组的第一个元素近似相等，精度为11位小数
        assert_array_almost_equal(
            bi[0],
            array([-1.173713222709127, -3.271093302836352, -4.830737841662016, -6.169852128310251, -7.376762079367764]),
            11,
        )

        # 断言 bi 的第二个元素与预期数组的第二个元素近似相等，精度为10位小数
        assert_array_almost_equal(
            bi[1],
            array([-2.294439682614122, -4.073155089071828, -5.512395729663599, -6.781294445990305, -7.940178689168587]),
            10,
        )

        # 断言 bi 的第三个元素与预期数组的第三个元素近似相等，精度为11位小数
        assert_array_almost_equal(
            bi[2],
            array([-0.454944383639657, 0.396522836094465, -0.367969161486959, 0.349499116831805, -0.336026240133662]),
            11,
        )

        # 断言 bi 的第四个元素与预期数组的第四个元素近似相等，精度为10位小数
        assert_array_almost_equal(
            bi[3],
            array([0.601957887976239, -0.760310141492801, 0.836991012619261, -0.88947990142654, 0.929983638568022]),
            10,
        )

    def test_ai_zeros(self):
        # 计算 special 模块中 ai_zeros 函数返回的结果

        # 获取 ai_zeros 函数返回的结果 ai
        ai = special.ai_zeros(1)
        # 断言 ai 与预期结果的近似相等
        assert_array_almost_equal(
            ai,
            (
                array([-2.33810741]),
                array([-1.01879297]),
                array([0.5357]),
                array([0.7012]),
            ),
            4,
        )

    @pytest.mark.fail_slow(5)
    # 定义一个测试函数，用于测试特殊数学函数库中的 AI 类型的零点
    def test_ai_zeros_big(self):
        # 调用特殊数学函数库中的 ai_zeros 函数，返回四个变量
        z, zp, ai_zpx, aip_zx = special.ai_zeros(50000)
        # 调用特殊数学函数库中的 airy 函数，计算 z 和 zp 的 Airy 函数值，并只保留前两个返回值
        ai_z, aip_z, _, _ = special.airy(z)
        ai_zp, aip_zp, _, _ = special.airy(zp)

        # 计算 AI 函数的包络线
        ai_envelope = 1/abs(z)**(1./4)
        aip_envelope = abs(zp)**(1./4)

        # 检查计算值是否相似
        assert_allclose(ai_zpx, ai_zp, rtol=1e-10)
        assert_allclose(aip_zx, aip_z, rtol=1e-10)

        # 检查 AI 函数是否在其包络线附近为零
        assert_allclose(ai_z/ai_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(aip_zp/aip_envelope, 0, atol=1e-10, rtol=0)

        # 检查前几个 AI 函数的零点，参考 DLMF 第 9.9.1 节
        assert_allclose(z[:6],
            [-2.3381074105, -4.0879494441, -5.5205598281,
             -6.7867080901, -7.9441335871, -9.0226508533], rtol=1e-10)
        assert_allclose(zp[:6],
            [-1.0187929716, -3.2481975822, -4.8200992112,
             -6.1633073556, -7.3721772550, -8.4884867340], rtol=1e-10)

    # 使用 pytest 的装饰器，标记此测试为慢速失败，当运行超过 5 秒时才会失败
    @pytest.mark.fail_slow(5)
    def test_bi_zeros_big(self):
        # 调用特殊数学函数库中的 bi_zeros 函数，返回四个变量
        z, zp, bi_zpx, bip_zx = special.bi_zeros(50000)
        # 调用特殊数学函数库中的 airy 函数，计算 z 和 zp 的 Airy 函数值，并只保留后两个返回值
        _, _, bi_z, bip_z = special.airy(z)
        _, _, bi_zp, bip_zp = special.airy(zp)

        # 计算 BI 函数的包络线
        bi_envelope = 1/abs(z)**(1./4)
        bip_envelope = abs(zp)**(1./4)

        # 检查计算值是否相似
        assert_allclose(bi_zpx, bi_zp, rtol=1e-10)
        assert_allclose(bip_zx, bip_z, rtol=1e-10)

        # 检查 BI 函数是否在其包络线附近为零
        assert_allclose(bi_z/bi_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(bip_zp/bip_envelope, 0, atol=1e-10, rtol=0)

        # 检查前几个 BI 函数的零点，参考 DLMF 第 9.9.2 节
        assert_allclose(z[:6],
            [-1.1737132227, -3.2710933028, -4.8307378417,
             -6.1698521283, -7.3767620794, -8.4919488465], rtol=1e-10)
        assert_allclose(zp[:6],
            [-2.2944396826, -4.0731550891, -5.5123957297,
             -6.7812944460, -7.9401786892, -9.0195833588], rtol=1e-10)
class TestAssocLaguerre:
    # 测试特殊函数 genlaguerre 和 assoc_laguerre 的功能
    def test_assoc_laguerre(self):
        # 调用 genlaguerre 生成 Laguerre 多项式
        a1 = special.genlaguerre(11,1)
        # 调用 assoc_laguerre 生成关联 Laguerre 函数
        a2 = special.assoc_laguerre(.2,11,1)
        # 断言两个函数在特定点的值近似相等
        assert_array_almost_equal(a2,a1(.2),8)
        # 再次调用 assoc_laguerre，不同的输入
        a2 = special.assoc_laguerre(1,11,1)
        # 断言两个函数在特定点的值近似相等
        assert_array_almost_equal(a2,a1(1),8)


class TestBesselpoly:
    # Bessel 多项式的测试类，暂无测试方法
    def test_besselpoly(self):
        pass


class TestKelvin:
    # 测试 Kelvin 函数族的各种函数
    def test_bei(self):
        # 调用 bei 函数计算 Modified Bessel 函数 I
        mbei = special.bei(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mbei, 0.9722916273066613, 5)  # this may not be exact

    def test_beip(self):
        # 调用 beip 函数计算 Modified Bessel 函数 I 的导数
        mbeip = special.beip(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mbeip, 0.91701361338403631, 5)  # this may not be exact

    def test_ber(self):
        # 调用 ber 函数计算 Modified Bessel 函数 K
        mber = special.ber(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mber, 0.75173418271380821, 5)  # this may not be exact

    def test_berp(self):
        # 调用 berp 函数计算 Modified Bessel 函数 K 的导数
        mberp = special.berp(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mberp, -0.49306712470943909, 5)  # this may not be exact

    def test_bei_zeros(self):
        # Abramowitz & Stegun, Table 9.12，计算 bei_zeros 的零点
        bi = special.bei_zeros(5)
        # 断言结果与已知值数组近似相等，精确到小数点后第四位
        assert_array_almost_equal(bi, array([5.02622,
                                             9.45541,
                                             13.89349,
                                             18.33398,
                                             22.77544]), 4)

    def test_beip_zeros(self):
        # 计算 beip_zeros 的零点
        bip = special.beip_zeros(5)
        # 断言结果与已知值数组近似相等，精确到小数点后第八位
        assert_array_almost_equal(bip, array([3.772673304934953,
                                              8.280987849760042,
                                              12.742147523633703,
                                              17.193431752512542,
                                              21.641143941167325]), 8)

    def test_ber_zeros(self):
        # 计算 ber_zeros 的零点
        ber = special.ber_zeros(5)
        # 断言结果与已知值数组近似相等，精确到小数点后第四位
        assert_array_almost_equal(ber, array([2.84892,
                                              7.23883,
                                              11.67396,
                                              16.11356,
                                              20.55463]), 4)

    def test_berp_zeros(self):
        # 计算 berp_zeros 的零点
        brp = special.berp_zeros(5)
        # 断言结果与已知值数组近似相等，精确到小数点后第四位
        assert_array_almost_equal(brp, array([6.03871,
                                              10.51364,
                                              14.96844,
                                              19.41758,
                                              23.86430]), 4)

    def test_kelvin(self):
        # 计算 Kelvin 函数的各种值
        mkelv = special.kelvin(2)
        # 断言结果数组中的各项值与已知 Kelvin 函数值近似相等，精确到小数点后第八位
        assert_array_almost_equal(mkelv, (special.ber(2) + special.bei(2)*1j,
                                          special.ker(2) + special.kei(2)*1j,
                                          special.berp(2) + special.beip(2)*1j,
                                          special.kerp(2) + special.keip(2)*1j), 8)

    def test_kei(self):
        # 调用 kei 函数计算 Kelvin 函数 I
        mkei = special.kei(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mkei, -0.20240006776470432, 5)

    def test_keip(self):
        # 调用 keip 函数计算 Kelvin 函数 I 的导数
        mkeip = special.keip(2)
        # 断言结果与已知值近似相等，精确到小数点后第五位
        assert_almost_equal(mkeip, 0.21980790991960536, 5)
    # 定义一个测试函数，用于测试 special 模块中的 ker 函数
    def test_ker(self):
        # 调用 special 模块中的 ker 函数，计算 mker 的值
        mker = special.ker(2)
        # 断言 mker 的值几乎等于 -0.041664513991509472，精确到小数点后第5位
        assert_almost_equal(mker, -0.041664513991509472, 5)

    # 定义一个测试函数，用于测试 special 模块中的 kerp 函数
    def test_kerp(self):
        # 调用 special 模块中的 kerp 函数，计算 mkerp 的值
        mkerp = special.kerp(2)
        # 断言 mkerp 的值几乎等于 -0.10660096588105264，精确到小数点后第5位
        assert_almost_equal(mkerp, -0.10660096588105264, 5)

    # 定义一个测试函数，用于测试 special 模块中的 kei_zeros 函数
    def test_kei_zeros(self):
        # 调用 special 模块中的 kei_zeros 函数，计算 kei 的值
        kei = special.kei_zeros(5)
        # 断言 kei 数组的值几乎等于给定数组，精确到小数点后第4位
        assert_array_almost_equal(kei, array([3.91467,
                                              8.34422,
                                              12.78256,
                                              17.22314,
                                              21.66464]), 4)

    # 定义一个测试函数，用于测试 special 模块中的 keip_zeros 函数
    def test_keip_zeros(self):
        # 调用 special 模块中的 keip_zeros 函数，计算 keip 的值
        keip = special.keip_zeros(5)
        # 断言 keip 数组的值几乎等于给定数组，精确到小数点后第4位
        assert_array_almost_equal(keip, array([4.93181,
                                               9.40405,
                                               13.85827,
                                               18.30717,
                                               22.75379]), 4)

    # 来自 A&S 书籍第9.9节的数字
    # 测试特殊函数 kelvin_zeros 的功能
    def test_kelvin_zeros(self):
        # 调用 kelvin_zeros 函数计算 Kelvin 函数的零点
        tmp = special.kelvin_zeros(5)
        # 将结果解包到多个变量中
        berz, beiz, kerz, keiz, berpz, beipz, kerpz, keipz = tmp
        # 检查 berz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(berz, array([2.84892,
                                               7.23883,
                                               11.67396,
                                               16.11356,
                                               20.55463]), 4)
        # 检查 beiz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(beiz, array([5.02622,
                                               9.45541,
                                               13.89349,
                                               18.33398,
                                               22.77544]), 4)
        # 检查 kerz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(kerz, array([1.71854,
                                               6.12728,
                                               10.56294,
                                               15.00269,
                                               19.44382]), 4)
        # 检查 keiz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(keiz, array([3.91467,
                                               8.34422,
                                               12.78256,
                                               17.22314,
                                               21.66464]), 4)
        # 检查 berpz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(berpz, array([6.03871,
                                                10.51364,
                                                14.96844,
                                                19.41758,
                                                23.86430]), 4)
        # 检查 beipz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(beipz, array([3.77267,
                                                8.28099,
                                                12.74215,
                                                17.19343,
                                                21.64114]), 4)
        # 检查 kerpz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(kerpz, array([2.66584,
                                                7.17212,
                                                11.63218,
                                                16.08312,
                                                20.53068]), 4)
        # 检查 keipz 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(keipz, array([4.93181,
                                                9.40405,
                                                13.85827,
                                                18.30717,
                                                22.75379]), 4)

    # 测试特殊函数 ker_zeros 的功能
    def test_ker_zeros(self):
        # 调用 ker_zeros 函数计算 Kelvin 函数的反函数的零点
        ker = special.ker_zeros(5)
        # 检查 ker 数组是否与给定的值数组几乎相等，精确到小数点后四位
        assert_array_almost_equal(ker, array([1.71854,
                                              6.12728,
                                              10.56294,
                                              15.00269,
                                              19.44381]), 4)
    # 定义测试函数 test_kerp_zeros(self)，用于测试 special 模块中的 kerp_zeros 函数
    def test_kerp_zeros(self):
        # 调用 special 模块中的 kerp_zeros 函数，返回长度为 5 的 Ker(r) 函数的零点列表
        kerp = special.kerp_zeros(5)
        # 使用断言检查 kerp 的值是否与预期值接近（保留四位小数）
        assert_array_almost_equal(kerp, array([2.66584,
                                               7.17212,
                                               11.63218,
                                               16.08312,
                                               20.53068]), 4)
class TestBernoulli:
    def test_bernoulli(self):
        # 调用特殊函数库中的伯努利数生成函数，生成前5个伯努利数的数组
        brn = special.bernoulli(5)
        # 断言生成的伯努利数数组与预期值几乎相等，精度为4位小数
        assert_array_almost_equal(brn, array([1.0000,
                                              -0.5000,
                                              0.1667,
                                              0.0000,
                                              -0.0333,
                                              0.0000]), 4)


class TestBeta:
    """
    Test beta and betaln.
    """

    def test_beta(self):
        # 断言特殊函数库中 beta 函数计算结果与预期值相等
        assert_equal(special.beta(1, 1), 1.0)
        # 断言特殊函数库中 beta 函数对于极端参数的近似值与 gamma 函数的计算结果几乎相等
        assert_allclose(special.beta(-100.3, 1e-200), special.gamma(1e-200))
        # 断言特殊函数库中 beta 函数计算结果与预期值在指定的相对和绝对误差范围内几乎相等
        assert_allclose(special.beta(0.0342, 171), 24.070498359873497,
                        rtol=1e-13, atol=0)

        # 计算 beta 函数的值
        bet = special.beta(2, 4)
        # 计算 gamma 函数值并根据 beta 函数的定义计算期望值
        betg = (special.gamma(2) * special.gamma(4)) / special.gamma(6)
        # 断言计算得到的 beta 函数值与期望值在指定的相对误差范围内几乎相等
        assert_allclose(bet, betg, rtol=1e-13)

    def test_beta_inf(self):
        # 断言特殊函数库中 beta 函数对于给定参数返回无穷大
        assert_(np.isinf(special.beta(-1, 2)))

    def test_betaln(self):
        # 断言特殊函数库中 betaln 函数计算结果与预期值相等
        assert_equal(special.betaln(1, 1), 0.0)
        # 断言特殊函数库中 betaln 函数对于极端参数的近似值与 gammaln 函数的计算结果几乎相等
        assert_allclose(special.betaln(-100.3, 1e-200),
                        special.gammaln(1e-200))
        # 断言特殊函数库中 betaln 函数计算结果与预期值在指定的相对和绝对误差范围内几乎相等
        assert_allclose(special.betaln(0.0342, 170), 3.1811881124242447,
                        rtol=1e-14, atol=0)

        # 计算 betaln 函数的值
        betln = special.betaln(2, 4)
        # 根据 beta 函数的定义计算其对数值
        bet = log(abs(special.beta(2, 4)))
        # 断言计算得到的 betaln 函数值与 beta 函数的对数值在指定的相对误差范围内几乎相等
        assert_allclose(betln, bet, rtol=1e-13)


class TestBetaInc:
    """
    Tests for betainc, betaincinv, betaincc, betainccinv.
    """

    def test_a1_b1(self):
        # 对于特定的参数值，betainc(1, 1, x) 应返回 x
        x = np.array([0, 0.25, 1])
        assert_equal(special.betainc(1, 1, x), x)
        assert_equal(special.betaincinv(1, 1, x), x)
        assert_equal(special.betaincc(1, 1, x), 1 - x)
        assert_equal(special.betainccinv(1, 1, x), 1 - x)

    # 参数化测试，使用不同的参数组合进行测试
    @pytest.mark.parametrize(
        'a, b, x, p',
        [(2, 4, 0.3138101704556974, 0.5),
         (0.0342, 171.0, 1e-10, 0.552699169018070910641),
         # gh-3761:
         (0.0342, 171, 8.42313169354797e-21, 0.25),
         # gh-4244:
         (0.0002742794749792665, 289206.03125, 1.639984034231756e-56,
          0.9688708782196045),
         # gh-12796:
         (4, 99997, 0.0001947841578892121, 0.999995)])
    def test_betainc_betaincinv(self, a, b, x, p):
        # 断言特殊函数库中 betainc 函数计算结果与预期概率 p 几乎相等
        p1 = special.betainc(a, b, x)
        assert_allclose(p1, p, rtol=1e-15)
        # 断言特殊函数库中 betaincinv 函数对于给定概率 p 的计算结果与预期输入值 x 几乎相等
        x1 = special.betaincinv(a, b, p)
        assert_allclose(x1, x, rtol=5e-13)

    # 预期值由 mpmath 计算得到：
    #    from mpmath import mp
    #    mp.dps = 100
    #    p = mp.betainc(a, b, x, 1, regularized=True)
    # 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数，测试特定的输入和预期输出
    @pytest.mark.parametrize('a, b, x, p',
                             [(2.5, 3.0, 0.25, 0.833251953125),
                              (7.5, 13.25, 0.375, 0.43298734645560368593),
                              (0.125, 7.5, 0.425, 0.0006688257851314237),
                              (0.125, 18.0, 1e-6, 0.72982359145096327654),
                              (0.125, 18.0, 0.996, 7.2745875538380150586e-46),
                              (0.125, 24.0, 0.75, 3.70853404816862016966e-17),
                              (16.0, 0.75, 0.99999999975,
                               5.4408759277418629909e-07),
                              # gh-4677 (numbers from stackoverflow question):
                              (0.4211959643503401, 16939.046996018118,
                               0.000815296167195521, 1e-7)])
    def test_betaincc_betainccinv(self, a, b, x, p):
        # 调用 special 模块中的 betaincc 函数，计算不完全贝塔函数值
        p1 = special.betaincc(a, b, x)
        # 使用 assert_allclose 检查计算结果 p1 是否与预期值 p 相近
        assert_allclose(p1, p, rtol=5e-15)
        # 调用 special 模块中的 betainccinv 函数，计算不完全贝塔逆函数值
        x1 = special.betainccinv(a, b, p)
        # 使用 assert_allclose 检查计算结果 x1 是否与预期值 x 相近
        assert_allclose(x1, x, rtol=8e-15)

    # 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数，测试极小 y 值情况下的计算
    @pytest.mark.parametrize(
        'a, b, y, ref',
        [(14.208308325339239, 14.208308325339239, 7.703145458496392e-307,
          8.566004561846704e-23),
         (14.0, 14.5, 1e-280, 2.9343915006642424e-21),
         (3.5, 15.0, 4e-95, 1.3290751429289227e-28),
         (10.0, 1.25, 2e-234, 3.982659092143654e-24),
         (4.0, 99997.0, 5e-88, 3.309800566862242e-27)]
    )
    def test_betaincinv_tiny_y(self, a, b, y, ref):
        # 测试极小 y 值情况下的 betaincinv 计算
        # 这个测试包含了对 boost 代码中问题的回归测试
        #
        # 参考值是使用 mpmath 计算得到的。例如，
        #
        #   from mpmath import mp
        #   mp.dps = 1000
        #   a = 14.208308325339239
        #   p = 7.703145458496392e-307
        #   x = mp.findroot(lambda t: mp.betainc(a, a, 0, t,
        #                                        regularized=True) - p,
        #                   x0=8.566e-23)
        #   print(float(x))
        #
        # 调用 special 模块中的 betaincinv 函数，计算不完全贝塔逆函数值
        x = special.betaincinv(a, b, y)
        # 使用 assert_allclose 检查计算结果 x 是否与预期值 ref 相近
        assert_allclose(x, ref, rtol=1e-14)

    # 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数，测试特定参数下 special 模块函数的异常情况
    @pytest.mark.parametrize('func', [special.betainc, special.betaincinv,
                                      special.betaincc, special.betainccinv])
    @pytest.mark.parametrize('args', [(-1.0, 2, 0.5), (0, 2, 0.5),
                                      (1.5, -2.0, 0.5), (1.5, 0, 0.5),
                                      (1.5, 2.0, -0.3), (1.5, 2.0, 1.1)])
    def test_betainc_domain_errors(self, func, args):
        # 使用 special.errstate 设置 domain='raise'，捕获特定的异常
        with special.errstate(domain='raise'):
            # 使用 pytest.raises 断言特定异常被抛出，匹配 'domain' 字符串
            with pytest.raises(special.SpecialFunctionError, match='domain'):
                # 调用 special 模块中的特定函数（通过参数化测试中的 func 参数传递），触发异常
                special.betainc(*args)
class TestCombinatorics:
    # 测试组合函数 special.comb 的不同用法

    def test_comb(self):
        # 测试组合函数对于多个参数的计算，并使用 assert_allclose 进行断言
        assert_allclose(special.comb([10, 10], [3, 4]), [120., 210.])
        # 测试组合函数对于单个参数的计算，并使用 assert_allclose 进行断言
        assert_allclose(special.comb(10, 3), 120.)
        # 测试组合函数对于单个参数的精确计算，并使用 assert_equal 进行断言
        assert_equal(special.comb(10, 3, exact=True), 120)
        # 测试组合函数对于重复组合的精确计算，并使用 assert_equal 进行断言
        assert_equal(special.comb(10, 3, exact=True, repetition=True), 220)

        # 使用列表推导式测试组合函数对于多个参数的精确计算，并使用 assert_allclose 进行断言
        assert_allclose([special.comb(20, k, exact=True) for k in range(21)],
                        special.comb(20, list(range(21))), atol=1e-15)

        # 测试特别大的整数参数的组合计算，并使用 assert_equal 进行断言
        ii = np.iinfo(int).max + 1
        assert_equal(special.comb(ii, ii-1, exact=True), ii)

        # 测试特定的组合数值，并使用 assert 进行断言
        expected = 100891344545564193334812497256
        assert special.comb(100, 50, exact=True) == expected

    # 测试组合函数对于 np.int64 类型参数的计算
    def test_comb_with_np_int64(self):
        n = 70
        k = 30
        np_n = np.int64(n)
        np_k = np.int64(k)
        # 测试组合函数对于 np.int64 类型参数的精确计算，并使用 assert 进行断言
        res_np = special.comb(np_n, np_k, exact=True)
        res_py = special.comb(n, k, exact=True)
        assert res_np == res_py

    # 测试组合函数处理特定情况下的零值，使用 assert_equal 进行断言
    def test_comb_zeros(self):
        assert_equal(special.comb(2, 3, exact=True), 0)
        assert_equal(special.comb(-1, 3, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=False), 0)
        # 测试组合函数对于包含非法参数的列表的处理，并使用 assert_allclose 进行断言
        assert_allclose(special.comb([2, -1, 2, 10], [3, 3, -1, 3]), [0., 0., 0., 120.])

    # 测试组合函数处理非整数参数的精确计算，使用 pytest 的 deprecated_call 进行断言
    def test_comb_exact_non_int_dep(self):
        msg = "`exact=True`"
        with pytest.deprecated_call(match=msg):
            special.comb(3.4, 4, exact=True)

    # 测试排列函数 special.perm 的不同用法
    def test_perm(self):
        assert_allclose(special.perm([10, 10], [3, 4]), [720., 5040.])
        assert_almost_equal(special.perm(10, 3), 720.)
        assert_equal(special.perm(10, 3, exact=True), 720)

    # 测试排列函数处理特定情况下的零值，使用 assert_equal 进行断言
    def test_perm_zeros(self):
        assert_equal(special.perm(2, 3, exact=True), 0)
        assert_equal(special.perm(-1, 3, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=False), 0)
        # 测试排列函数对于包含非法参数的列表的处理，并使用 assert_allclose 进行断言
        assert_allclose(special.perm([2, -1, 2, 10], [3, 3, -1, 3]), [0., 0., 0., 720.])

    # 测试排列函数处理特定情况下的错误信息，使用 pytest 的 raises 进行断言
    def test_perm_iv(self):
        # 测试排列函数对于非标量参数使用时的错误信息处理
        with pytest.raises(ValueError, match="scalar integers"):
            special.perm([1, 2], [4, 5], exact=True)

        # 测试排列函数对于非整数参数使用时的 deprecated_call 断言
        with pytest.deprecated_call(match="Non-integer"):
            special.perm(4.6, 6, exact=True)
        with pytest.deprecated_call(match="Non-integer"):
            special.perm(-4.6, 3, exact=True)
        with pytest.deprecated_call(match="Non-integer"):
            special.perm(4, -3.9, exact=True)

        # 测试排列函数对于未支持的非整数参数使用时的 raises 断言
        with pytest.raises(ValueError, match="Non-integer"):
            special.perm(6.0, 4.6, exact=True)
    # 定义测试函数，用于测试 special 模块中的 cbrt 函数
    def test_cbrt(self):
        # 调用 special 模块中的 cbrt 函数计算 27 的立方根
        cb = special.cbrt(27)
        # 计算 27 的立方根作为参考值
        cbrl = 27**(1.0/3.0)
        # 断言 cb 和 cbrl 大致相等
        assert_approx_equal(cb, cbrl)

    # 定义测试函数，用于测试 special 模块中的 cbrt 函数（更大的输入值）
    def test_cbrtmore(self):
        # 调用 special 模块中的 cbrt 函数计算 27.9 的立方根
        cb1 = special.cbrt(27.9)
        # 计算 27.9 的立方根作为参考值
        cbrl1 = 27.9**(1.0/3.0)
        # 断言 cb1 和 cbrl1 大致相等，精确到小数点后第 8 位
        assert_almost_equal(cb1, cbrl1, 8)

    # 定义测试函数，用于测试 special 模块中的 cosdg 函数
    def test_cosdg(self):
        # 调用 special 模块中的 cosdg 函数计算 90 度的余弦值（度数制）
        cdg = special.cosdg(90)
        # 计算 90 度的余弦值作为参考值
        cdgrl = cos(pi/2.0)
        # 断言 cdg 和 cdgrl 大致相等，精确到小数点后第 8 位
        assert_almost_equal(cdg, cdgrl, 8)

    # 定义测试函数，用于测试 special 模块中的 cosdg 函数（更多测试）
    def test_cosdgmore(self):
        # 调用 special 模块中的 cosdg 函数计算 30 度的余弦值（度数制）
        cdgm = special.cosdg(30)
        # 计算 30 度的余弦值作为参考值
        cdgmrl = cos(pi/6.0)
        # 断言 cdgm 和 cdgmrl 大致相等，精确到小数点后第 8 位
        assert_almost_equal(cdgm, cdgmrl, 8)

    # 定义测试函数，用于测试 special 模块中的 cosm1 函数
    def test_cosm1(self):
        # 调用 special 模块中的 cosm1 函数分别计算 0、0.3 和 pi/10 的余弦值减去 1
        cs = (special.cosm1(0), special.cosm1(.3), special.cosm1(pi/10))
        # 计算 0、0.3 和 pi/10 的余弦值减去 1作为参考值
        csrl = (cos(0)-1, cos(.3)-1, cos(pi/10)-1)
        # 断言 cs 和 csrl 的每个元素大致相等，精确到小数点后第 8 位
        assert_array_almost_equal(cs, csrl, 8)

    # 定义测试函数，用于测试 special 模块中的 cotdg 函数
    def test_cotdg(self):
        # 调用 special 模块中的 cotdg 函数计算 30 度的余切值（度数制）
        ct = special.cotdg(30)
        # 计算 30 度的余切值的倒数作为参考值
        ctrl = tan(pi/6.0)**(-1)
        # 断言 ct 和 ctrl 大致相等，精确到小数点后第 8 位
        assert_almost_equal(ct, ctrl, 8)

    # 定义测试函数，用于测试 special 模块中的 cotdg 函数（更多测试）
    def test_cotdgmore(self):
        # 调用 special 模块中的 cotdg 函数计算 45 度的余切值（度数制）
        ct1 = special.cotdg(45)
        # 计算 45 度的余切值的倒数作为参考值
        ctrl1 = tan(pi/4.0)**(-1)
        # 断言 ct1 和 ctrl1 大致相等，精确到小数点后第 8 位
        assert_almost_equal(ct1, ctrl1, 8)

    # 定义测试函数，用于测试 special 模块中的 cotdg 函数在特定角度上的值
    def test_specialpoints(self):
        # 断言特定角度上调用 special 模块中的 cotdg 函数计算结果，与预期值大致相等，精确到小数点后第 14 位
        assert_almost_equal(special.cotdg(45), 1.0, 14)
        assert_almost_equal(special.cotdg(-45), -1.0, 14)
        assert_almost_equal(special.cotdg(90), 0.0, 14)
        assert_almost_equal(special.cotdg(-90), 0.0, 14)
        assert_almost_equal(special.cotdg(135), -1.0, 14)
        assert_almost_equal(special.cotdg(-135), 1.0, 14)
        assert_almost_equal(special.cotdg(225), 1.0, 14)
        assert_almost_equal(special.cotdg(-225), -1.0, 14)
        assert_almost_equal(special.cotdg(270), 0.0, 14)
        assert_almost_equal(special.cotdg(-270), 0.0, 14)
        assert_almost_equal(special.cotdg(315), -1.0, 14)
        assert_almost_equal(special.cotdg(-315), 1.0, 14)
        assert_almost_equal(special.cotdg(765), 1.0, 14)

    # 定义测试函数，用于测试 special 模块中的 sinc 函数
    def test_sinc(self):
        # 断言调用 special 模块中的 sinc 函数计算结果，与预期值相等
        assert_array_equal(special.sinc([0]), 1)
        assert_equal(special.sinc(0.0), 1.0)

    # 定义测试函数，用于测试 special 模块中的 sindg 函数
    def test_sindg(self):
        # 调用 special 模块中的 sindg 函数计算 90 度的正弦值（度数制）
        sn = special.sindg(90)
        # 断言 sn 和 1.0 相等
        assert_equal(sn, 1.0)

    # 定义测试函数，用于测试 special 模块中的 sindg 函数（更多测试）
    def test_sindgmore(self):
        # 调用 special 模块中的 sindg 函数计算 30 度的正弦值（度数制）
        snm = special.sindg(30)
        # 计算 30 度的正弦值作为参考值
        snmrl = sin(pi/6.0)
        # 断言 snm 和 snmrl 大致相等，精确到小数点后第 8 位
        assert_almost_equal(snm, snmrl, 8)
        # 调用 special 模块中的 sindg 函数计算 45 度的正弦值（度数制）
        snm1 = special.sindg(45)
        # 计算 45 度的正弦值作为参考值
        snmrl1 = sin(pi/4.0)
        # 断言 snm1 和 snmrl1 大致相等，精确到小数点后第 8 位
        assert_almost_equal(snm1, snmrl1, 8)
class TestTandg:

    def test_tandg(self):
        # 调用特殊函数库中的 tandg 函数，计算角度为 30 度的正切值
        tn = special.tandg(30)
        # 使用标准数学库中的 tan 函数，计算角度为 pi/6.0 弧度的正切值
        tnrl = tan(pi/6.0)
        # 断言近似相等，精确到小数点后第 8 位
        assert_almost_equal(tn, tnrl, 8)

    def test_tandgmore(self):
        # 调用特殊函数库中的 tandg 函数，计算角度为 45 度的正切值
        tnm = special.tandg(45)
        # 使用标准数学库中的 tan 函数，计算角度为 pi/4.0 弧度的正切值
        tnmrl = tan(pi/4.0)
        # 断言近似相等，精确到小数点后第 8 位
        assert_almost_equal(tnm, tnmrl, 8)
        # 调用特殊函数库中的 tandg 函数，计算角度为 60 度的正切值
        tnm1 = special.tandg(60)
        # 使用标准数学库中的 tan 函数，计算角度为 pi/3.0 弧度的正切值
        tnmrl1 = tan(pi/3.0)
        # 断言近似相等，精确到小数点后第 8 位
        assert_almost_equal(tnm1, tnmrl1, 8)

    def test_specialpoints(self):
        # 断言特殊角度的正切值近似为 0，精确到小数点后第 14 位
        assert_almost_equal(special.tandg(0), 0.0, 14)
        assert_almost_equal(special.tandg(45), 1.0, 14)
        assert_almost_equal(special.tandg(-45), -1.0, 14)
        assert_almost_equal(special.tandg(135), -1.0, 14)
        assert_almost_equal(special.tandg(-135), 1.0, 14)
        assert_almost_equal(special.tandg(180), 0.0, 14)
        assert_almost_equal(special.tandg(-180), 0.0, 14)
        assert_almost_equal(special.tandg(225), 1.0, 14)
        assert_almost_equal(special.tandg(-225), -1.0, 14)
        assert_almost_equal(special.tandg(315), -1.0, 14)
        assert_almost_equal(special.tandg(-315), 1.0, 14)


class TestEllip:

    def test_ellipj_nan(self):
        """Regression test for #912."""
        # 调用特殊函数库中的 ellipj 函数，验证输入参数为 NaN 的情况是否能正常处理

    def test_ellipj(self):
        # 调用特殊函数库中的 ellipj 函数，计算参数 (0.2, 0) 的椭圆函数值
        el = special.ellipj(0.2, 0)
        # 预期的结果列表
        rel = [sin(0.2), cos(0.2), 1.0, 0.20]
        # 断言数组几乎相等，精确到小数点后第 13 位
        assert_array_almost_equal(el, rel, 13)

    def test_ellipk(self):
        # 调用特殊函数库中的 ellipk 函数，计算参数 0.2 的完全椭圆积分 K(k) 的值
        elk = special.ellipk(.2)
        # 断言近似相等，精确到小数点后第 11 位
        assert_almost_equal(elk, 1.659623598610528, 11)

        # 断言特殊情况下的 ellipkm1 函数返回值
        assert_equal(special.ellipkm1(0.0), np.inf)
        assert_equal(special.ellipkm1(1.0), pi/2)
        assert_equal(special.ellipkm1(np.inf), 0.0)
        assert_equal(special.ellipkm1(np.nan), np.nan)
        assert_equal(special.ellipkm1(-1), np.nan)
        assert_allclose(special.ellipk(-10), 0.7908718902387385)
    # 定义一个测试方法 test_ellipkinc，用于测试特殊函数库中的 ellipkinc 函数
    def test_ellipkinc(self):
        # 调用 ellipkinc 函数计算特定参数下的椭圆积分值
        elkinc = special.ellipkinc(pi/2, .2)
        # 调用 ellipk 函数计算相同参数下的椭圆积分值
        elk = special.ellipk(0.2)
        # 使用 assert_almost_equal 断言，验证两个计算结果的精度
        assert_almost_equal(elkinc, elk, 15)
        
        # 设置角度和角度转换后的弧度值
        alpha = 20 * pi / 180
        phi = 45 * pi / 180
        # 计算 sin(alpha) 的平方
        m = sin(alpha)**2
        # 再次调用 ellipkinc 函数，计算另一组参数下的椭圆积分值
        elkinc = special.ellipkinc(phi, m)
        # 使用 assert_almost_equal 断言，验证计算结果与预期值的接近程度
        assert_almost_equal(elkinc, 0.79398143, 8)
        
        # 执行注释：从《A & S》的第614页获取的测试用例
        # 使用 assert_equal 断言，验证特定参数下的 ellipkinc 函数值
        assert_equal(special.ellipkinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipkinc(pi/2, 1.0), np.inf)
        assert_equal(special.ellipkinc(pi/2, -np.inf), 0.0)
        assert_equal(special.ellipkinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipkinc(pi/2, 2), np.nan)
        assert_equal(special.ellipkinc(0, 0.5), 0.0)
        assert_equal(special.ellipkinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipkinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipkinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipkinc(np.nan, np.nan), np.nan)
        
        # 使用 assert_allclose 断言，验证特定参数下的 ellipkinc 函数值与预期值的接近程度
        assert_allclose(special.ellipkinc(0.38974112035318718, 1), 0.4, rtol=1e-14)
        assert_allclose(special.ellipkinc(1.5707, -10), 0.79084284661724946)

    # 定义第二个测试方法 test_ellipkinc_2，用于测试特殊情况下的 ellipkinc 函数
    def test_ellipkinc_2(self):
        # 进行回归测试，检测问题编号为 gh-3550 的 bug 是否修复
        # 当 m 参数传入一个极其接近但小于 0.68359375000000011 的值时，ellipkinc 函数返回 NaN
        mbad = 0.68359375000000011
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)  # 使用 np.nextafter 函数获取略小于 mbad 的浮点数
        mvals = []
        # 创建一个包含多个 m 值的列表，这些值略大于前一个值，用于测试多次调用的结果
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        # 调用 ellipkinc 函数，计算给定 phi 和 mvals 的椭圆积分值
        f = special.ellipkinc(phi, mvals)
        # 使用 assert_array_almost_equal_nulp 断言，验证计算结果与预期值的接近度
        assert_array_almost_equal_nulp(f, np.full_like(f, 1.0259330100195334), 1)
        
        # 对于特定的 phi + n * pi，再次测试相同的问题，确保修复在这些情况下仍有效
        f1 = special.ellipkinc(phi + pi, mvals)
        # 使用 assert_array_almost_equal_nulp 断言，验证计算结果与预期值的接近度
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 5.1296650500976675), 2)
    def test_ellipkinc_singular(self):
        # 测试特殊情况下的特殊椭圆积分函数 ellipkinc(phi, 1)
        # phi 只有在 (-pi/2, pi/2) 范围内时，其值有闭合形式且有限
        xlog = np.logspace(-300, -17, 25)  # 生成对数间隔的数组，用于测试
        xlin = np.linspace(1e-17, 0.1, 25)  # 生成线性间隔的数组，用于测试
        xlin2 = np.linspace(0.1, pi/2, 25, endpoint=False)  # 生成线性间隔的数组，用于测试

        # 使用 assert_allclose 函数比较特殊椭圆积分函数的值和其数值近似值
        assert_allclose(special.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)),
                        rtol=1e14)
        # 使用 assert_equal 函数比较特殊椭圆积分函数在极限值处的返回值
        assert_equal(special.ellipkinc(np.pi/2, 1), np.inf)
        assert_allclose(special.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)),
                        rtol=1e14)
        assert_equal(special.ellipkinc(-np.pi/2, 1), np.inf)

    def test_ellipe(self):
        ele = special.ellipe(.2)  # 计算第二类不完全椭圆积分函数的值
        assert_almost_equal(ele, 1.4890350580958529, 8)

        # 使用 assert_equal 函数比较第二类不完全椭圆积分函数在特定输入下的返回值
        assert_equal(special.ellipe(0.0), pi/2)
        assert_equal(special.ellipe(1.0), 1.0)
        assert_equal(special.ellipe(-np.inf), np.inf)
        assert_equal(special.ellipe(np.nan), np.nan)
        assert_equal(special.ellipe(2), np.nan)
        # 使用 assert_allclose 函数比较第二类不完全椭圆积分函数的数值近似值
        assert_allclose(special.ellipe(-10), 3.6391380384177689)

    def test_ellipeinc(self):
        eleinc = special.ellipeinc(pi/2, .2)  # 计算第一类不完全椭圆积分函数的值
        ele = special.ellipe(0.2)
        assert_almost_equal(eleinc, ele, 14)
        # 根据 A & S 的第 617 页进行测试
        alpha, phi = 52*pi/180, 35*pi/180
        m = sin(alpha)**2
        # 计算第一类不完全椭圆积分函数的值
        eleinc = special.ellipeinc(phi, m)
        assert_almost_equal(eleinc, 0.58823065, 8)

        # 使用 assert_equal 函数比较第一类不完全椭圆积分函数在特定输入下的返回值
        assert_equal(special.ellipeinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipeinc(pi/2, 1.0), 1.0)
        assert_equal(special.ellipeinc(pi/2, -np.inf), np.inf)
        assert_equal(special.ellipeinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipeinc(pi/2, 2), np.nan)
        assert_equal(special.ellipeinc(0, 0.5), 0.0)
        assert_equal(special.ellipeinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipeinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipeinc(np.inf, -np.inf), np.inf)
        assert_equal(special.ellipeinc(-np.inf, -np.inf), -np.inf)
        assert_equal(special.ellipeinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(np.nan, 0.5), np.nan)
        # 使用 assert_allclose 函数比较第一类不完全椭圆积分函数的数值近似值
        assert_allclose(special.ellipeinc(1.5707, -10), 3.6388185585822876)
    def test_ellipeinc_2(self):
        # 定义一个测试函数，用于检查 gh-3550 的回归问题
        # 当 ellipeinc(phi, mbad) 返回 NaN，并且 mvals[2:6] 的值是正确值的两倍时
        mbad = 0.68359375000000011  # 设置一个有问题的浮点数 mbad
        phi = 0.9272952180016123  # 设置 phi 的值
        m = np.nextafter(mbad, 0)  # 使用 np.nextafter 将 mbad 微调为接近零的值，赋给 m
        mvals = []  # 初始化一个空列表 mvals
        for j in range(10):  # 开始一个循环，迭代 10 次
            mvals.append(m)  # 将当前的 m 值添加到 mvals 列表中
            m = np.nextafter(m, 1)  # 使用 np.nextafter 将 m 微调为更大的值
        f = special.ellipeinc(phi, mvals)  # 调用特殊函数库中的 ellipeinc 函数，传入 phi 和 mvals，计算结果并赋给 f
        assert_array_almost_equal_nulp(f, np.full_like(f, 0.84442884574781019), 2)
        # 使用 assert_array_almost_equal_nulp 函数断言 f 应接近于填充值为 0.84442884574781019 的数组，精度为 2
        # 这个 bug 在 phi + n * pi (其中 n 是小整数) 时也可能出现
        f1 = special.ellipeinc(phi + pi, mvals)
        # 调用 ellipeinc 函数，传入 phi + pi 和 mvals，计算结果并赋给 f1
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 3.3471442287390509), 4)
        # 使用 assert_array_almost_equal_nulp 函数断言 f1 应接近于填充值为 3.3471442287390509 的数组，精度为 4
class TestEllipCarlson:
    """Test for Carlson elliptic integrals ellipr[cdfgj].
    The special values used in these tests can be found in Sec. 3 of Carlson
    (1994), https://arxiv.org/abs/math/9409227
    """

    # 测试 elliprc 函数的方法
    def test_elliprc(self):
        # 断言 elliprc(1, 1) 接近于 1
        assert_allclose(elliprc(1, 1), 1)
        # 断言 elliprc(1, inf) 等于 0.0
        assert elliprc(1, inf) == 0.0
        # 断言 elliprc(1, 0) 返回 NaN
        assert isnan(elliprc(1, 0))
        # 断言 elliprc(1, complex(1, inf)) 等于 0.0
        assert elliprc(1, complex(1, inf)) == 0.0

        # 准备测试参数数组
        args = array([[0.0, 0.25],
                      [2.25, 2.0],
                      [0.0, 1.0j],
                      [-1.0j, 1.0j],
                      [0.25, -2.0],
                      [1.0j, -1.0]])
        # 准备预期结果数组
        expected_results = array([np.pi,
                                  np.log(2.0),
                                  1.1107207345396 * (1.0-1.0j),
                                  1.2260849569072-0.34471136988768j,
                                  np.log(2.0) / 3.0,
                                  0.77778596920447+0.19832484993429j])
        # 遍历参数数组并断言 elliprc 函数返回值接近于预期结果
        for i, arr in enumerate(args):
            assert_allclose(elliprc(*arr), expected_results[i])

    # 测试 elliprd 函数的方法
    def test_elliprd(self):
        # 断言 elliprd(1, 1, 1) 接近于 1
        assert_allclose(elliprd(1, 1, 1), 1)
        # 断言 elliprd(0, 2, 1) / 3.0 接近于 0.59907011736779610371
        assert_allclose(elliprd(0, 2, 1) / 3.0, 0.59907011736779610371)
        # 断言 elliprd(1, 1, inf) 等于 0.0
        assert elliprd(1, 1, inf) == 0.0
        # 断言 elliprd(1, 1, 0) 返回正无穷
        assert np.isinf(elliprd(1, 1, 0))
        # 断言 elliprd(1, 1, complex(0, 0)) 返回正无穷
        assert np.isinf(elliprd(1, 1, complex(0, 0)))
        # 断言 elliprd(0, 1, complex(0, 0)) 返回正无穷
        assert np.isinf(elliprd(0, 1, complex(0, 0)))
        # 断言 elliprd(1, 1, -np.finfo(np.float64).tiny / 2.0) 返回 NaN
        assert isnan(elliprd(1, 1, -np.finfo(np.float64).tiny / 2.0))
        # 断言 elliprd(1, 1, complex(-1, 0)) 返回 NaN
        assert isnan(elliprd(1, 1, complex(-1, 0)))

        # 准备测试参数数组
        args = array([[0.0, 2.0, 1.0],
                      [2.0, 3.0, 4.0],
                      [1.0j, -1.0j, 2.0],
                      [0.0, 1.0j, -1.0j],
                      [0.0, -1.0+1.0j, 1.0j],
                      [-2.0-1.0j, -1.0j, -1.0+1.0j]])
        # 准备预期结果数组
        expected_results = array([1.7972103521034,
                                  0.16510527294261,
                                  0.65933854154220,
                                  1.2708196271910+2.7811120159521j,
                                  -1.8577235439239-0.96193450888839j,
                                  1.8249027393704-1.2218475784827j])
        # 遍历参数数组并断言 elliprd 函数返回值接近于预期结果
        for i, arr in enumerate(args):
            assert_allclose(elliprd(*arr), expected_results[i])
    def test_elliprf(self):
        # 测试函数 elliprf，验证其返回值是否接近预期值
        assert_allclose(elliprf(1, 1, 1), 1)
        # 验证 elliprf 对于不同参数的返回值是否接近预期值
        assert_allclose(elliprf(0, 1, 2), 1.31102877714605990523)
        # 验证当一个参数为无穷大时，elliprf 的返回值是否为 0.0
        assert elliprf(1, inf, 1) == 0.0
        # 验证 elliprf 返回值是否为无穷大
        assert np.isinf(elliprf(0, 1, 0))
        # 验证 elliprf 对于参数中包含负数时是否返回 NaN
        assert isnan(elliprf(1, 1, -1))
        # 验证 elliprf 返回值是否为 0.0，其中一个参数为复数无穷大
        assert elliprf(complex(inf), 0, 1) == 0.0
        # 验证 elliprf 返回值是否为 NaN，其中一个参数为复数负无穷大
        assert isnan(elliprf(1, 1, complex(-inf, 1)))
        
        # 创建包含不同参数组合的数组
        args = array([[1.0, 2.0, 0.0],
                      [1.0j, -1.0j, 0.0],
                      [0.5, 1.0, 0.0],
                      [-1.0+1.0j, 1.0j, 0.0],
                      [2.0, 3.0, 4.0],
                      [1.0j, -1.0j, 2.0],
                      [-1.0+1.0j, 1.0j, 1.0-1.0j]])
        # 创建期望的结果数组
        expected_results = array([1.3110287771461,
                                  1.8540746773014,
                                  1.8540746773014,
                                  0.79612586584234-1.2138566698365j,
                                  0.58408284167715,
                                  1.0441445654064,
                                  0.93912050218619-0.53296252018635j])
        # 遍历参数数组，验证 elliprf 返回值是否接近期望结果
        for i, arr in enumerate(args):
            assert_allclose(elliprf(*arr), expected_results[i])

    def test_elliprg(self):
        # 测试函数 elliprg，验证其返回值是否接近预期值
        assert_allclose(elliprg(1, 1, 1), 1)
        # 验证 elliprg 对于不同参数的返回值是否接近预期值
        assert_allclose(elliprg(0, 0, 1), 0.5)
        assert_allclose(elliprg(0, 0, 0), 0)
        # 验证 elliprg 返回值是否为无穷大，其中一个参数为无穷大
        assert np.isinf(elliprg(1, inf, 1))
        # 验证 elliprg 返回值是否为无穷大，其中一个参数为复数无穷大
        assert np.isinf(elliprg(complex(inf), 1, 1))
        
        # 创建包含不同参数组合的数组
        args = array([[0.0, 16.0, 16.0],
                      [2.0, 3.0, 4.0],
                      [0.0, 1.0j, -1.0j],
                      [-1.0+1.0j, 1.0j, 0.0],
                      [-1.0j, -1.0+1.0j, 1.0j],
                      [0.0, 0.0796, 4.0]])
        # 创建期望的结果数组
        expected_results = array([np.pi,
                                  1.7255030280692,
                                  0.42360654239699,
                                  0.44660591677018+0.70768352357515j,
                                  0.36023392184473+0.40348623401722j,
                                  1.0284758090288])
        # 遍历参数数组，验证 elliprg 返回值是否接近期望结果
        for i, arr in enumerate(args):
            assert_allclose(elliprg(*arr), expected_results[i])
    # 定义一个测试方法 test_elliprj，用于测试 elliprj 函数的不同输入情况
    def test_elliprj(self):
        # 断言：验证 elliprj(1, 1, 1, 1) 的返回值接近 1
        assert_allclose(elliprj(1, 1, 1, 1), 1)
        # 断言：验证 elliprj(1, 1, inf, 1) 的返回值为 0.0
        assert elliprj(1, 1, inf, 1) == 0.0
        # 断言：验证 elliprj(1, 0, 0, 0) 的返回值是否为 NaN
        assert isnan(elliprj(1, 0, 0, 0))
        # 断言：验证 elliprj(-1, 1, 1, 1) 的返回值是否为 NaN
        assert isnan(elliprj(-1, 1, 1, 1))
        # 断言：验证 elliprj(1, 1, 1, inf) 的返回值为 0.0
        assert elliprj(1, 1, 1, inf) == 0.0
        # 准备一个包含不同参数组合的数组 args，用于循环测试
        args = array([[0.0, 1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0, 5.0],
                      [2.0, 3.0, 4.0, -1.0+1.0j],
                      [1.0j, -1.0j, 0.0, 2.0],
                      [-1.0+1.0j, -1.0-1.0j, 1.0, 2.0],
                      [1.0j, -1.0j, 0.0, 1.0-1.0j],
                      [-1.0+1.0j, -1.0-1.0j, 1.0, -3.0+1.0j],
                      [2.0, 3.0, 4.0, -0.5],    # 柯西主值
                      [2.0, 3.0, 4.0, -5.0]])   # 柯西主值
        # 准备一个预期结果的数组 expected_results，与 args 数组对应
        expected_results = array([0.77688623778582,
                                  0.14297579667157,
                                  0.13613945827771-0.38207561624427j,
                                  1.6490011662711,
                                  0.94148358841220,
                                  1.8260115229009+1.2290661908643j,
                                  -0.61127970812028-1.0684038390007j,
                                  0.24723819703052,    # 柯西主值
                                  -0.12711230042964])  # 柯西主值
        # 对每个参数组合进行测试，验证 elliprj 函数的返回结果是否接近预期结果
        for i, arr in enumerate(args):
            assert_allclose(elliprj(*arr), expected_results[i])

    # 使用 pytest 的标记 xfail，声明以下测试预计会失败，原因是在 32 位系统上精度不足
    @pytest.mark.xfail(reason="Insufficient accuracy on 32-bit")
    # 定义一个测试方法 test_elliprj_hard，测试 elliprj 函数在极端情况下的精确性
    def test_elliprj_hard(self):
        # 断言：验证 elliprj 函数在给定参数下的返回值，与预期值在指定的相对误差和绝对误差范围内匹配
        assert_allclose(elliprj(6.483625725195452e-08,
                                1.1649136528196886e-27,
                                3.6767340167168e+13,
                                0.493704617023468),
                        8.63426920644241857617477551054e-6,
                        rtol=5e-15, atol=1e-20)
        # 断言：验证 elliprj 函数在给定参数下的返回值，与预期值在指定的相对误差和绝对误差范围内匹配
        assert_allclose(elliprj(14.375105857849121,
                                9.993988969725365e-11,
                                1.72844262269944e-26,
                                5.898871222598245e-06),
                        829774.1424801627252574054378691828,
                        rtol=5e-15, atol=1e-20)
class TestEllipLegendreCarlsonIdentities:
    """Test identities expressing the Legendre elliptic integrals in terms
    of Carlson's symmetric integrals.  These identities can be found
    in the DLMF https://dlmf.nist.gov/19.25#i .
    """

    def setup_class(self):
        # 创建包含从 -1 到 1，间隔为 0.01 的 numpy 数组
        self.m_n1_1 = np.arange(-1., 1., 0.01)
        # 对于双精度浮点数，这是 -(2**1024)
        self.max_neg = finfo(double).min
        # 大量非常负的数值
        self.very_neg_m = -1. * 2.**arange(-1 +
                                           np.log2(-self.max_neg), 0.,
                                           -1.)
        # 将 max_neg、very_neg_m 和 m_n1_1 合并成一个 numpy 数组
        self.ms_up_to_1 = np.concatenate(([self.max_neg],
                                          self.very_neg_m,
                                          self.m_n1_1))

    def test_k(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        """
        m = self.ms_up_to_1
        # 检查 ellipk(m) 和 elliprf(0., 1.-m, 1.) 是否非常接近
        assert_allclose(ellipk(m), elliprf(0., 1.-m, 1.))

    def test_km1(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        But with the ellipkm1 function
        """
        # 对于双精度浮点数，这是 2**-1022
        tiny = finfo(double).tiny
        # 所有这些小的 2 的幂，直到 2**-1
        m1 = tiny * 2.**arange(0., -np.log2(tiny))
        # 检查 ellipkm1(m1) 和 elliprf(0., m1, 1.) 是否非常接近
        assert_allclose(ellipkm1(m1), elliprf(0., m1, 1.))

    def test_e(self):
        """Test identity:
        E(m) = 2*R_G(0, 1-k^2, 1)
        """
        m = self.ms_up_to_1
        # 检查 ellipe(m) 和 2.*elliprg(0., 1.-m, 1.) 是否非常接近
        assert_allclose(ellipe(m), 2.*elliprg(0., 1.-m, 1.))


class TestErf:

    def test_erf(self):
        # 计算特定值的误差函数 erf(.25)
        er = special.erf(.25)
        assert_almost_equal(er, 0.2763263902, 8)

    def test_erf_zeros(self):
        # 计算前 5 个误差函数的零点
        erz = special.erf_zeros(5)
        erzr = array([1.45061616+1.88094300j,
                     2.24465928+2.61657514j,
                     2.83974105+3.17562810j,
                     3.33546074+3.64617438j,
                     3.76900557+4.06069723j])
        # 检查计算结果与预期结果的数组是否非常接近
        assert_array_almost_equal(erz, erzr, 4)

    def _check_variant_func(self, func, other_func, rtol, atol=0):
        np.random.seed(1234)
        n = 10000
        x = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        y = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        z = x + 1j*y

        with np.errstate(all='ignore'):
            w = other_func(z)
            w_real = other_func(x).real

            mask = np.isfinite(w)
            w = w[mask]
            z = z[mask]

            mask = np.isfinite(w_real)
            w_real = w_real[mask]
            x = x[mask]

            # 测试实数和复数变体函数的一致性
            assert_func_equal(func, w, z, rtol=rtol, atol=atol)
            assert_func_equal(func, w_real, x, rtol=rtol, atol=atol)

    def test_erfc_consistent(self):
        # 检查误差函数的互补函数是否一致
        self._check_variant_func(
            cephes.erfc,
            lambda z: 1 - cephes.erf(z),
            rtol=1e-12,
            atol=1e-14  # <- 测试函数失去精度
            )
    # 测试特殊函数 erfcx 的一致性
    def test_erfcx_consistent(self):
        # 使用 _check_variant_func 方法检查 erfcx 的变体函数
        self._check_variant_func(
            cephes.erfcx,  # 调用 cephes 库中的 erfcx 函数
            lambda z: np.exp(z*z) * cephes.erfc(z),  # 使用 lambda 表达式定义变体函数
            rtol=1e-12  # 定义相对容差
            )

    # 测试特殊函数 erfi 的一致性
    def test_erfi_consistent(self):
        # 使用 _check_variant_func 方法检查 erfi 的变体函数
        self._check_variant_func(
            cephes.erfi,  # 调用 cephes 库中的 erfi 函数
            lambda z: -1j * cephes.erf(1j*z),  # 使用 lambda 表达式定义变体函数
            rtol=1e-12  # 定义相对容差
            )

    # 测试特殊函数 dawsn 的一致性
    def test_dawsn_consistent(self):
        # 使用 _check_variant_func 方法检查 dawsn 的变体函数
        self._check_variant_func(
            cephes.dawsn,  # 调用 cephes 库中的 dawsn 函数
            lambda z: sqrt(pi)/2 * np.exp(-z*z) * cephes.erfi(z),  # 使用 lambda 表达式定义变体函数
            rtol=1e-12  # 定义相对容差
            )

    # 测试特殊函数 erf 对于 NaN 和 Inf 输入的处理
    def test_erf_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -1, 1]
        # 使用 assert_allclose 断言函数检查特殊函数 erf 的输出是否符合预期
        assert_allclose(special.erf(vals), expected, rtol=1e-15)

    # 测试特殊函数 erfc 对于 NaN 和 Inf 输入的处理
    def test_erfc_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, 2, 0]
        # 使用 assert_allclose 断言函数检查特殊函数 erfc 的输出是否符合预期
        assert_allclose(special.erfc(vals), expected, rtol=1e-15)

    # 测试特殊函数 erfcx 对于 NaN 和 Inf 输入的处理
    def test_erfcx_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, np.inf, 0]
        # 使用 assert_allclose 断言函数检查特殊函数 erfcx 的输出是否符合预期
        assert_allclose(special.erfcx(vals), expected, rtol=1e-15)

    # 测试特殊函数 erfi 对于 NaN 和 Inf 输入的处理
    def test_erfi_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -np.inf, np.inf]
        # 使用 assert_allclose 断言函数检查特殊函数 erfi 的输出是否符合预期
        assert_allclose(special.erfi(vals), expected, rtol=1e-15)

    # 测试特殊函数 dawsn 对于 NaN 和 Inf 输入的处理
    def test_dawsn_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -0.0, 0.0]
        # 使用 assert_allclose 断言函数检查特殊函数 dawsn 的输出是否符合预期
        assert_allclose(special.dawsn(vals), expected, rtol=1e-15)

    # 测试特殊函数 wofz 对于 NaN 和 Inf 输入的处理
    def test_wofz_nan_inf(self):
        # 定义输入值和预期输出值列表
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan + np.nan * 1.j, 0.-0.j, 0.+0.j]
        # 使用 assert_allclose 断言函数检查特殊函数 wofz 的输出是否符合预期
        assert_allclose(special.wofz(vals), expected, rtol=1e-15)
class TestEuler:
    # 定义测试 Euler 常数的测试类
    def test_euler(self):
        # 计算 Euler 常数 euler(0), euler(1), euler(2)，并检查是否引发段错误
        eu0 = special.euler(0)
        eu1 = special.euler(1)
        eu2 = special.euler(2)   # just checking segfaults
        # 断言 euler(0) 应该接近 [1]
        assert_allclose(eu0, [1], rtol=1e-15)
        # 断言 euler(1) 应该接近 [1, 0]
        assert_allclose(eu1, [1, 0], rtol=1e-15)
        # 断言 euler(2) 应该接近 [1, 0, -1]
        assert_allclose(eu2, [1, 0, -1], rtol=1e-15)
        
        # 计算 euler(24)，与 MathWorld 中预期的值进行比较
        eu24 = special.euler(24)
        mathworld = [1,1,5,61,1385,50521,2702765,199360981,
                     19391512145,2404879675441,
                     370371188237525,69348874393137901,
                     15514534163557086905]
        # 创建一个空数组，准备存储计算出的正确值
        correct = zeros((25,),'d')
        # 遍历 mathworld 中的值
        for k in range(0,13):
            if (k % 2):
                # 如果 k 是奇数，则取负数
                correct[2*k] = -float(mathworld[k])
            else:
                # 如果 k 是偶数，则取正数
                correct[2*k] = float(mathworld[k])
        
        # 使用 numpy 的错误状态管理，计算相对误差并将 NaN 转换为 0
        with np.errstate(all='ignore'):
            err = nan_to_num((eu24-correct)/correct)
            # 计算最大的相对误差
            errmax = max(err)
        
        # 断言最大相对误差应该接近 0，精度为 14
        assert_almost_equal(errmax, 0.0, 14)


class TestExp:
    # 定义测试指数函数的测试类
    def test_exp2(self):
        # 计算 2 的幂次方
        ex = special.exp2(2)
        exrl = 2**2
        # 断言计算结果与预期结果相等
        assert_equal(ex, exrl)

    def test_exp2more(self):
        # 计算 2.5 的幂次方
        exm = special.exp2(2.5)
        exmrl = 2**(2.5)
        # 断言计算结果与预期结果在精度为 8 时几乎相等
        assert_almost_equal(exm, exmrl, 8)

    def test_exp10(self):
        # 计算 10 的幂次方
        ex = special.exp10(2)
        exrl = 10**2
        # 断言计算结果与预期结果在相对误差允许范围内接近
        assert_approx_equal(ex, exrl)

    def test_exp10more(self):
        # 计算 10 的幂次方（浮点数）
        exm = special.exp10(2.5)
        exmrl = 10**(2.5)
        # 断言计算结果与预期结果在精度为 8 时几乎相等
        assert_almost_equal(exm, exmrl, 8)

    def test_expm1(self):
        # 计算 exp(x)-1 的值，其中 x 分别为 2, 3, 4
        ex = (special.expm1(2), special.expm1(3), special.expm1(4))
        exrl = (exp(2)-1, exp(3)-1, exp(4)-1)
        # 断言计算结果与预期结果在精度为 8 时几乎相等
        assert_array_almost_equal(ex, exrl, 8)

    def test_expm1more(self):
        # 计算 exp(x)-1 的值，其中 x 分别为 2, 2.1, 2.2
        ex1 = (special.expm1(2), special.expm1(2.1), special.expm1(2.2))
        exrl1 = (exp(2)-1, exp(2.1)-1, exp(2.2)-1)
        # 断言计算结果与预期结果在精度为 8 时几乎相等
        assert_array_almost_equal(ex1, exrl1, 8)


class TestFactorialFunctions:
    # 测试阶乘函数的测试类
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_scalar_return_type(self, exact):
        # 断言 factorial, factorial2, factorialk 返回的结果都是标量
        assert np.isscalar(special.factorial(1, exact=exact))
        assert np.isscalar(special.factorial2(1, exact=exact))
        assert np.isscalar(special.factorialk(1, 3, exact=exact))

    @pytest.mark.parametrize("n", [-1, -2, -3])
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_negative(self, exact, n):
        # 断言对于负整数 n，factorial, factorial2, factorialk 的结果都应该是 0
        assert_equal(special.factorial(n, exact=exact), 0)
        assert_equal(special.factorial2(n, exact=exact), 0)
        assert_equal(special.factorialk(n, 3, exact=exact), 0)

    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_negative_array(self, exact):
        # 测试 factorial, factorial2, factorialk 对于数组输入的行为
        assert_func = assert_array_equal if exact else assert_allclose
        # 对于 n < 0，期望的输出都是 [0, 0, 1, 1]
        assert_func(special.factorial([-5, -4, 0, 1], exact=exact),
                    [0, 0, 1, 1])
        assert_func(special.factorial2([-5, -4, 0, 1], exact=exact),
                    [0, 0, 1, 1])
        assert_func(special.factorialk([-5, -4, 0, 1], 3, exact=exact),
                    [0, 0, 1, 1])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("content", [np.nan, None, np.datetime64('nat')],
                             ids=["NaN", "None", "NaT"])
    def test_factorialx_nan(self, content, exact):
        # 测试特殊情况下的 factorial、factorial2 和 factorialk 函数
    
        # 对于标量输入
        assert special.factorial(content, exact=exact) is np.nan
        assert special.factorial2(content, exact=exact) is np.nan
        assert special.factorialk(content, 3, exact=exact) is np.nan
    
        # 对于类似数组的输入（初始化为默认 dtype 的 np.array）
        if content is not np.nan:
            # None 导致对象 dtype，不支持；datetime 也不支持
            with pytest.raises(ValueError, match="Unsupported datatype.*"):
                special.factorial([content], exact=exact)
        elif exact:
            with pytest.raises(ValueError, match="factorial with `exact=Tr.*"):
                special.factorial([content], exact=exact)
        else:
            # 返回的结果应该是 NaN
            assert np.isnan(special.factorial([content], exact=exact)[0])
    
        # factorial2 和 factorialk 不支持数组情况，因为存在 dtype 约束
        with pytest.raises(ValueError, match="factorial2 does not support.*"):
            special.factorial2([content], exact=exact)
        with pytest.raises(ValueError, match="factorialk does not support.*"):
            special.factorialk([content], 3, exact=exact)
    
        # 数组情况也在 test_factorial{,2,k}_corner_cases 中进行测试
    
    @pytest.mark.parametrize("levels", range(1, 5))
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_array_shape(self, levels, exact):
        # 测试 factorial、factorial2 和 factorialk 函数在数组形状下的行为
    
        def _nest_me(x, k=1):
            """
            Double x and nest it k times
    
            For example:
            >>> _nest_me([3, 4], 2)
            [[[3, 4], [3, 4]], [[3, 4], [3, 4]]]
            """
            if k == 0:
                return x
            else:
                return _nest_me([x, x], k-1)
    
        def _check(res, nucleus):
            # 期望的结果，通过递归生成与 levels 对应的嵌套结构
            exp = np.array(_nest_me(nucleus, k=levels), dtype=object)
            # 检查 ndarray 的形状是否保持不变
            # 由于 numpy/numpy#21220，需要将结果转换为 float
            assert_allclose(res.astype(np.float64), exp.astype(np.float64))
    
        # 生成嵌套数组 n
        n = np.array(_nest_me([5, 25], k=levels))
    
        # 期望的核心值
        exp_nucleus = {1: [120, math.factorial(25)],
                       # factorial2() 和 factorialk() 的正确性在其他地方测试
                       2: [15, special.factorial2(25, exact=True)],
                       3: [10, special.factorialk(25, 3, exact=True)]}
    
        # 测试 factorial、factorial2 和 factorialk 函数在数组形状下的表现
        _check(special.factorial(n, exact=exact), exp_nucleus[1])
        _check(special.factorial2(n, exact=exact), exp_nucleus[2])
        _check(special.factorialk(n, 3, exact=exact), exp_nucleus[3])
    # 定义一个测试函数，用于测试特定维度和数据类型的阶乘计算
    def test_factorialx_array_dimension(self, dim, dtype, exact):
        # 创建一个包含单个元素 5 的 numpy 数组，数据类型为 dtype，维度为 dim
        n = np.array(5, dtype=dtype, ndmin=dim)
        # 预期的阶乘结果字典，键为 1、2、3 分别对应不同的阶乘函数结果
        exp = {1: 120, 2: 15, 3: 10}
        # 使用 assert_allclose 函数检查 special.factorial 的计算结果是否与预期一致
        assert_allclose(special.factorial(n, exact=exact),
                        np.array(exp[1], ndmin=dim))
        # 使用 assert_allclose 函数检查 special.factorial2 的计算结果是否与预期一致
        assert_allclose(special.factorial2(n, exact=exact),
                        np.array(exp[2], ndmin=dim))
        # 使用 assert_allclose 函数检查 special.factorialk 的计算结果是否与预期一致
        assert_allclose(special.factorialk(n, 3, exact=exact),
                        np.array(exp[3], ndmin=dim))

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("level", range(1, 5))
    # 定义一个参数化测试函数，用于测试不同级别和精确度的阶乘计算
    def test_factorialx_array_like(self, level, exact):
        # 定义一个递归函数 _nest_me，用于生成嵌套列表
        def _nest_me(x, k=1):
            if k == 0:
                return x
            else:
                return _nest_me([x], k-1)

        # 使用 _nest_me 生成一个嵌套列表，作为阶乘函数的输入参数 n
        n = _nest_me([5], k=level-1)  # nested list
        # 预期的阶乘结果字典，键为 1、2、3 分别对应不同的阶乘函数结果
        exp_nucleus = {1: 120, 2: 15, 3: 10}
        # 根据 exact 的值选择使用 assert_array_equal 或 assert_allclose 进行断言
        assert_func = assert_array_equal if exact else assert_allclose
        # 使用 assert_func 函数检查 special.factorial 的计算结果是否与预期一致
        assert_func(special.factorial(n, exact=exact),
                    np.array(exp_nucleus[1], ndmin=level))
        # 使用 assert_func 函数检查 special.factorial2 的计算结果是否与预期一致
        assert_func(special.factorial2(n, exact=exact),
                    np.array(exp_nucleus[2], ndmin=level))
        # 使用 assert_func 函数检查 special.factorialk 的计算结果是否与预期一致
        assert_func(special.factorialk(n, 3, exact=exact),
                    np.array(exp_nucleus[3], ndmin=level))

    # 标注 n=170 是最后一个能够使得 factorial(n) 的结果适合 float64 的整数
    @pytest.mark.parametrize('n', range(30, 180, 10))
    # 定义一个测试函数，用于测试阶乘计算的精确度
    def test_factorial_accuracy(self, n):
        # 比较 exact=True 和 False 的计算结果，即精确度是否超过指定的容差值
        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        # 使用 assert_allclose 函数比较精确计算结果和近似计算结果是否一致
        assert_allclose(float(special.factorial(n, exact=True)),
                        special.factorial(n, exact=False), rtol=rtol)
        # 使用 assert_allclose 函数比较精确计算结果和近似计算结果是否一致，需要将精确结果转换为 float 类型
        assert_allclose(special.factorial([n], exact=True).astype(float),
                        special.factorial([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(30, 180, 10)))
    # 定义一个测试函数，用于比较阶乘计算结果与 math.factorial 的结果
    def test_factorial_int_reference(self, n):
        # 比较所有结果与 math.factorial 的结果是否一致
        correct = math.factorial(n)
        # 使用 assert_array_equal 函数比较计算结果和正确结果是否一致
        assert_array_equal(correct, special.factorial(n, True))
        # 使用 assert_array_equal 函数比较计算结果和正确结果是否一致，需要将返回的数组结果与单个数值进行比较
        assert_array_equal(correct, special.factorial([n], True)[0])

        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        # 使用 assert_allclose 函数比较精确计算结果和近似计算结果是否一致
        assert_allclose(float(correct), special.factorial(n, False),
                        rtol=rtol)
        # 使用 assert_allclose 函数比较精确计算结果和近似计算结果是否一致，需要将返回的数组结果与单个数值进行比较
        assert_allclose(float(correct), special.factorial([n], False)[0],
                        rtol=rtol)
    # 定义一个测试函数，用于验证阶乘计算对于浮点数的准确性
    def test_factorial_float_reference(self):
        # 定义内部函数 `_check`，用于检查阶乘计算的准确性
        def _check(n, expected):
            # 使用 `special.factorial` 计算阶乘并断言结果与预期值接近
            assert_allclose(special.factorial(n), expected)
            # 对单个数值或数组进行阶乘计算，并断言结果与预期值接近
            assert_allclose(special.factorial([n])[0], expected)
            # 使用 `exact=True` 参数进行浮点数阶乘计算会引发错误，验证是否正确引发 ValueError
            with pytest.raises(ValueError, match="Non-integer values.*"):
                assert_allclose(special.factorial(n, exact=True), expected)
            # 使用 `exact=True` 参数对数组进行浮点数阶乘计算会引发错误，验证是否正确引发 ValueError
            with pytest.raises(ValueError, match="factorial with `exact=Tr.*"):
                special.factorial([n], exact=True)

        # 使用 mpmath 提供的参考值进行阶乘计算的准确性验证
        _check(0.01, 0.994325851191506032181932988)
        _check(1.11, 1.051609009483625091514147465)
        _check(5.55, 314.9503192327208241614959052)
        _check(11.1, 50983227.84411615655137170553)
        _check(33.3, 2.493363339642036352229215273e+37)
        _check(55.5, 9.479934358436729043289162027e+73)
        _check(77.7, 3.060540559059579022358692625e+114)
        _check(99.9, 5.885840419492871504575693337e+157)
        # 接近 float64 的最大值进行阶乘计算的准确性验证
        _check(170.6243, 1.79698185749571048960082e+308)

    # 使用 pytest 的参数化标记定义多个测试参数
    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    # 使用 pytest 的参数化标记定义多种内容的数组进行测试
    # 包括空数组、非空数组、包含 NaN 和混合内容的数组
    @pytest.mark.parametrize("content",
                             [[], [1], [1.1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[1.1]", "[NaN]", "[NaN, 1]"])
    # 定义测试方法，用于测试阶乘函数在特殊情况下的行为
    def test_factorial_array_corner_cases(self, content, dim, exact, dtype):
        # 如果数据类型是 np.int64 并且内容中有 NaN，则跳过测试
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        
        # 如果 dim 大于 0 或者 content 的长度不为 1，则不改变 content；否则将 content[0] 赋给 content
        content = content if (dim > 0 or len(content) != 1) else content[0]
        
        # 根据给定的 dim 和 dtype 创建一个至少为 dim 维的数组 n
        n = np.array(content, ndmin=dim, dtype=dtype)
        
        result = None
        
        # 如果 content 为空
        if not content:
            # 计算阶乘，结果存储在 result 中
            result = special.factorial(n, exact=exact)
        
        # 如果 n 的 dtype 不是整数类型或浮点数类型
        elif not (np.issubdtype(n.dtype, np.integer)
                  or np.issubdtype(n.dtype, np.floating)):
            # 检查是否抛出 ValueError 异常，匹配字符串为 "Unsupported datatype*"
            with pytest.raises(ValueError, match="Unsupported datatype*"):
                special.factorial(n, exact=exact)
        
        # 如果要求精确计算并且 n 的 dtype 不是整数类型
        elif exact and not np.issubdtype(n.dtype, np.integer):
            # 检查是否抛出 ValueError 异常，匹配字符串为 "factorial with `exact=.*"
            with pytest.raises(ValueError, match="factorial with `exact=.*"):
                special.factorial(n, exact=exact)
        
        else:
            # 否则，调用阶乘函数计算结果，存储在 result 中
            result = special.factorial(n, exact=exact)
        
        # assert_equal 函数无法区分标量和相同值的 0 维数组，
        # 参考 https://github.com/numpy/numpy/issues/24050
        def assert_really_equal(x, y):
            assert type(x) == type(y), f"types not equal: {type(x)}, {type(y)}"
            assert_equal(x, y)
        
        # 如果 result 不为空
        if result is not None:
            # 如果 n 是 0 维数组，则 n_flat 为 n.ravel()；否则 n_flat 为 n
            n_flat = n.ravel() if n.ndim else n
            # 如果 n 的大小不为 0，则 ref 存储计算后的阶乘结果；否则为一个空列表
            ref = special.factorial(n_flat, exact=exact) if n.size else []
            # 期望的结果为空列表当且仅当 n 为空，并且与 n 具有相同的 dtype 和维度
            expected = np.array(ref, ndmin=dim, dtype=dtype)
            # 断言 result 和 expected 相等
            assert_really_equal(result, expected)

    # 使用 pytest 的参数化功能，对精确计算进行测试
    @pytest.mark.parametrize("exact", [True, False])
    # 使用 pytest 的参数化功能，对不同的 n 进行测试，包括整数、浮点数、复数、NaN 和 None
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    # 测试阶乘函数在标量特殊情况下的行为
    def test_factorial_scalar_corner_cases(self, n, exact):
        # 如果 n 是 None 或者 NaN，或者 n 是整数类型或浮点数类型
        if (n is None or n is np.nan or np.issubdtype(type(n), np.integer)
                or np.issubdtype(type(n), np.floating)):
            # 如果 n 是浮点数类型并且要求精确计算，并且 n 不是 NaN
            if (np.issubdtype(type(n), np.floating) and exact
                    and n is not np.nan):
                # 检查是否抛出 ValueError 异常，匹配字符串为 "Non-integer values.*"
                with pytest.raises(ValueError, match="Non-integer values.*"):
                    special.factorial(n, exact=exact)
            else:
                # 否则，计算阶乘，结果存储在 result 中
                result = special.factorial(n, exact=exact)
                # 如果 n 是 NaN 或者 None，则 exp 设置为 np.nan；否则设置为计算后的阶乘结果
                exp = np.nan if n is np.nan or n is None else special.factorial(n)
                # 断言 result 和 exp 相等
                assert_equal(result, exp)
        else:
            # 如果 n 的类型不被支持，抛出 ValueError 异常，匹配字符串为 "Unsupported datatype*"
            with pytest.raises(ValueError, match="Unsupported datatype*"):
                special.factorial(n, exact=exact)

    # 使用奇数增量确保测试奇数和偶数的情况！
    @pytest.mark.parametrize('n', range(30, 180, 11))
    def test_factorial2_accuracy(self, n):
        # 测试函数：test_factorial2_accuracy
        # 对比 exact=True 和 False 的情况，即精确度是否优于指定的容差值。

        rtol = 2e-14 if sys.platform == 'win32' else 1e-15
        # 根据操作系统平台设置容差值
        assert_allclose(float(special.factorial2(n, exact=True)),
                        special.factorial2(n, exact=False), rtol=rtol)
        # 检查使用 exact=True 和 False 时的计算结果是否在容差范围内
        assert_allclose(special.factorial2([n], exact=True).astype(float),
                        special.factorial2([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(30, 180, 11)))
    def test_factorial2_int_reference(self, n):
        # 测试函数：test_factorial2_int_reference
        # 比较所有情况下的正确值

        # 由于溢出问题，不能使用 np.product
        correct = functools.reduce(operator.mul, list(range(n, 0, -2)), 1)

        # 检查使用 exact=True 时的计算结果是否与正确值相等
        assert_array_equal(correct, special.factorial2(n, True))
        assert_array_equal(correct, special.factorial2([n], True)[0])

        # 检查使用 exact=False 时的计算结果是否在容差范围内
        assert_allclose(float(correct), special.factorial2(n, False))
        assert_allclose(float(correct), special.factorial2([n], False)[0])

    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    # 测试空数组和非空数组，包括 NaN 和混合内容
    @pytest.mark.parametrize("content", [[], [1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[NaN]", "[NaN, 1]"])
    def test_factorial2_array_corner_cases(self, content, dim, exact, dtype):
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        # 如果 content 是 0 维数组，则 np.array(x, ndim=0) 不会是 0 维。除非 x 也是
        content = content if (dim > 0 or len(content) != 1) else content[0]
        # 创建指定 dtype 和维度的 numpy 数组
        n = np.array(content, ndmin=dim, dtype=dtype)
        if np.issubdtype(n.dtype, np.integer) or (not content):
            # 没有错误
            result = special.factorial2(n, exact=exact)
            # 对于 exact=True 或空数组，预期结果与 n 相同；否则在容差范围内
            func = assert_equal if exact or (not content) else assert_allclose
            func(result, n)
        else:
            # 预期抛出 ValueError 错误
            with pytest.raises(ValueError, match="factorial2 does not*"):
                special.factorial2(n, 3)

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    def test_factorial2_scalar_corner_cases(self, n, exact):
        # 检查 n 是否为 None、NaN 或整数类型
        if n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            # 如果符合条件，调用 special.factorial2 计算阶乘
            result = special.factorial2(n, exact=exact)
            # 如果 n 是 NaN 或 None，则期望结果为 NaN，否则为 special.factorial(n)
            exp = np.nan if n is np.nan or n is None else special.factorial(n)
            # 断言计算结果与期望结果相等
            assert_equal(result, exp)
        else:
            # 如果 n 不满足条件，预期会引发 ValueError 异常
            with pytest.raises(ValueError, match="factorial2 does not*"):
                special.factorial2(n, exact=exact)

    @pytest.mark.parametrize("k", range(1, 5))
    # 注意：n=170 是最后一个使得 factorial(n) 在 float64 范围内的整数；
    # 使用奇数增量以确保测试奇数和偶数
    @pytest.mark.parametrize('n', range(170, 20, -29))
    def test_factorialk_accuracy(self, n, k):
        # 比较 exact=True 和 False 的情况，即近似精度是否优于指定的容差值

        # 需要将精确结果转换为 float，因为受 numpy/numpy#21220 影响
        assert_allclose(float(special.factorialk(n, k=k, exact=True)),
                        special.factorialk(n, k=k, exact=False))
        assert_allclose(special.factorialk([n], k=k, exact=True).astype(float),
                        special.factorialk([n], k=k, exact=False))

    @pytest.mark.parametrize('k', list(range(1, 5)) + [10, 20])
    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(22, 100, 11)))
    def test_factorialk_int_reference(self, n, k):
        # 比较所有的值与正确的参考值

        # 可以使用 np.product 会更好，但在 Windows 上存在问题，参见 numpy/numpy#21219
        correct = functools.reduce(operator.mul, list(range(n, 0, -k)), 1)

        # 断言数组的内容与正确值相等
        assert_array_equal(correct, special.factorialk(n, k, True))
        assert_array_equal(correct, special.factorialk([n], k, True)[0])

        # 比较浮点数的近似值与正确值
        assert_allclose(float(correct), special.factorialk(n, k, False))
        assert_allclose(float(correct), special.factorialk([n], k, False)[0])

    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    # 测试空数组和非空数组，包括 NaN 和混合内容
    @pytest.mark.parametrize("content", [[], [1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[NaN]", "[NaN, 1]"])
    # 测试阶乘函数对数组参数的边界情况处理
    def test_factorialk_array_corner_cases(self, content, dim, exact, dtype):
        # 如果数据类型是 np.int64 并且内容中包含 NaN，则跳过测试
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        
        # 如果 dim > 0 或者 content 的长度不为 1，则不修改 content
        # 否则将 content 赋值为 content[0]
        content = content if (dim > 0 or len(content) != 1) else content[0]
        
        # 创建一个 numpy 数组 n，确保至少为 dim 维，并根据 exact 决定数据类型
        n = np.array(content, ndmin=dim, dtype=dtype if exact else np.float64)
        
        # 如果 n 的数据类型是整数类型，或者 content 为空
        if np.issubdtype(n.dtype, np.integer) or (not content):
            # 断言 factorialk 函数对 n 的计算结果与 n 相等
            assert_equal(special.factorialk(n, 3, exact=exact), n)
        else:
            # 否则预期会引发 ValueError 异常，且异常信息包含 "factorialk does not"
            with pytest.raises(ValueError, match="factorialk does not*"):
                special.factorialk(n, 3, exact=exact)

    # 使用不同参数组合对阶乘函数的标量参数进行边界情况测试
    @pytest.mark.parametrize("exact", [True, False, None])
    @pytest.mark.parametrize("k", range(1, 5))
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    def test_factorialk_scalar_corner_cases(self, n, k, exact):
        # 如果 n 为 None、np.nan 或者是整数类型，则执行以下条件语句
        if n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            # 如果 exact 为 None，则使用 pytest.deprecated_call 检查警告信息
            if exact is None:
                with pytest.deprecated_call(match="factorialk will default.*"):
                    result = special.factorialk(n, k=k, exact=exact)
            else:
                # 否则直接计算 factorialk 函数的结果
                result = special.factorialk(n, k=k, exact=exact)

            # 检查结果是否为 np.nan，如果 n 是 np.nan 或 None，则预期结果也为 np.nan，否则为 1
            nan_cond = n is np.nan or n is None
            expected = np.nan if nan_cond else 1
            assert_equal(result, expected)
        else:
            # 否则预期会引发 ValueError 异常，且异常信息包含 "factorialk does not"
            with pytest.raises(ValueError, match="factorialk does not*"):
                # 使用 suppress_warnings 过滤 DeprecationWarning
                with suppress_warnings() as sup:
                    sup.filter(DeprecationWarning, "factorialk will default")
                    special.factorialk(n, k=k, exact=exact)

    # 测试阶乘函数对 k 参数的异常情况处理
    @pytest.mark.parametrize("k", [0, 1.1, np.nan, "1"])
    def test_factorialk_raises_k(self, k):
        # 预期会引发 ValueError 异常，且异常信息包含 "k must be a positive integer"
        with pytest.raises(ValueError, match="k must be a positive integer*"):
            special.factorialk(1, k)

    # 使用不同参数组合对阶乘函数的 k 参数进行测试
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("k", range(1, 12))
    # 定义测试函数，用于测试特殊的 factorialk 函数的数据类型
    def test_factorialk_dtype(self, k, exact):
        # 构建参数字典
        kw = {"k": k, "exact": exact}
        # 如果 exact 为 True，并且 k 在 64 位整数的 factorialk 限制中
        if exact and k in _FACTORIALK_LIMITS_64BITS.keys():
            # 使用 32 位整数限制下的 k 构造 numpy 数组 n
            n = np.array([_FACTORIALK_LIMITS_32BITS[k]])
            # 断言特定条件下 factorialk 的返回类型为 np_long
            assert_equal(special.factorialk(n, **kw).dtype, np_long)
            # 断言 factorialk(n + 1, **kw) 的返回类型为 np.int64
            assert_equal(special.factorialk(n + 1, **kw).dtype, np.int64)
            # 断言 factorialk(n + 1, **kw) 大于 np.int32 的最大值，以确保极限性
            assert special.factorialk(n + 1, **kw) > np.iinfo(np.int32).max

            # 使用 64 位整数限制下的 k 构造 numpy 数组 n
            n = np.array([_FACTORIALK_LIMITS_64BITS[k]])
            # 断言 factorialk(n, **kw) 的返回类型为 np.int64
            assert_equal(special.factorialk(n, **kw).dtype, np.int64)
            # 断言 factorialk(n + 1, **kw) 的返回类型为 object
            assert_equal(special.factorialk(n + 1, **kw).dtype, object)
            # 断言 factorialk(n + 1, **kw) 大于 np.int64 的最大值，以确保极限性
            assert special.factorialk(n + 1, **kw) > np.iinfo(np.int64).max
        else:
            # 对于不在 64 位整数限制中的 k，使用默认值 1
            n = np.array([_FACTORIALK_LIMITS_64BITS.get(k, 1)])
            # 根据 exact 参数确定返回的 dtype 类型
            # 当 exact=True 且 k >= 10 时，始终返回 object 类型
            # 当 exact=False 时，始终返回 np.float64 类型
            dtype = object if exact else np.float64
            assert_equal(special.factorialk(n, **kw).dtype, dtype)

    # 定义测试函数，用于测试 factorial 函数处理混合 NaN 输入的情况
    def test_factorial_mixed_nan_inputs(self):
        # 构造包含 NaN 的 numpy 数组 x
        x = np.array([np.nan, 1, 2, 3, np.nan])
        # 预期的输出结果，NaN 不变，其它值分别计算阶乘
        expected = np.array([np.nan, 1, 2, 6, np.nan])
        # 断言 factorial 函数处理 exact=False 的情况下的输出符合预期
        assert_equal(special.factorial(x, exact=False), expected)
        # 使用 pytest 断言，在 exact=True 情况下，factorial 函数抛出 ValueError 异常
        with pytest.raises(ValueError, match="factorial with `exact=True.*"):
            special.factorial(x, exact=True)
class TestFresnel:
    @pytest.mark.parametrize("z, s, c", [
        # 一些正值
        (.5, 0.064732432859999287, 0.49234422587144644),
        (.5 + .0j, 0.064732432859999287, 0.49234422587144644),
        # 负半环
        # https://github.com/scipy/scipy/issues/12309
        # 参考值可通过以下链接进行验证
        # https://www.wolframalpha.com/input/?i=FresnelS%5B-2.0+%2B+0.1i%5D
        # https://www.wolframalpha.com/input/?i=FresnelC%5B-2.0+%2B+0.1i%5D
        (
            -2.0 + 0.1j,
            -0.3109538687728942-0.0005870728836383176j,
            -0.4879956866358554+0.10670801832903172j
        ),
        (
            -0.1 - 1.5j,
            -0.03918309471866977+0.7197508454568574j,
            0.09605692502968956-0.43625191013617465j
        ),
        # "大" 值会触发不同的算法，即 |z| >= 4.5,
        # 确保测试浮点数和复数值；会使用不同的算法
        (6.0, 0.44696076, 0.49953147),
        (6.0 + 0.0j, 0.44696076, 0.49953147),
        (6.0j, -0.44696076j, 0.49953147j),
        (-6.0 + 0.0j, -0.44696076, -0.49953147),
        (-6.0j, 0.44696076j, -0.49953147j),
        # 无穷大
        (np.inf, 0.5, 0.5),
        (-np.inf, -0.5, -0.5),
    ])
    def test_fresnel_values(self, z, s, c):
        # 使用 special 模块中的 fresnel 函数计算 Fresnel 积分
        frs = array(special.fresnel(z))
        # 断言计算结果准确到小数点后第八位
        assert_array_almost_equal(frs, array([s, c]), 8)

    # 来自 A & S 第 329 页表 7.11 的值
    # 第四位小数略有修正
    def test_fresnel_zeros(self):
        # 获取 Fresnel 零点
        szo, czo = special.fresnel_zeros(5)
        # 断言计算结果准确到小数点后第三位
        assert_array_almost_equal(szo,
                                  array([2.0093+0.2885j,
                                          2.8335+0.2443j,
                                          3.4675+0.2185j,
                                          4.0026+0.2009j,
                                          4.4742+0.1877j]), 3)
        assert_array_almost_equal(czo,
                                  array([1.7437+0.3057j,
                                          2.6515+0.2529j,
                                          3.3204+0.2240j,
                                          3.8757+0.2047j,
                                          4.3611+0.1907j]), 3)
        # 计算 Fresnel 积分的值
        vals1 = special.fresnel(szo)[0]
        vals2 = special.fresnel(czo)[1]
        # 断言计算结果准确到小数点后第十四位
        assert_array_almost_equal(vals1, 0, 14)
        assert_array_almost_equal(vals2, 0, 14)

    def test_fresnelc_zeros(self):
        # 获取 Fresnel 零点
        szo, czo = special.fresnel_zeros(6)
        # 使用 special 模块中的 fresnelc_zeros 函数计算 FresnelC 零点
        frc = special.fresnelc_zeros(6)
        # 断言两者的近似相等性，精确到小数点后第十二位
        assert_array_almost_equal(frc, czo, 12)

    def test_fresnels_zeros(self):
        # 获取 Fresnel 零点
        szo, czo = special.fresnel_zeros(5)
        # 使用 special 模块中的 fresnels_zeros 函数计算 FresnelS 零点
        frs = special.fresnels_zeros(5)
        # 断言两者的近似相等性，精确到小数点后第十二位
        assert_array_almost_equal(frs, szo, 12)


class TestGamma:
    def test_gamma(self):
        # 使用 special 模块中的 gamma 函数计算 Gamma 函数值
        gam = special.gamma(5)
        # 断言 Gamma 函数值为 24.0
        assert_equal(gam, 24.0)
    # 测试特殊函数 gammaln 的正确性
    def test_gammaln(self):
        # 计算 gamma 函数的自然对数值
        gamln = special.gammaln(3)
        # 计算 gamma 函数的对数值
        lngam = log(special.gamma(3))
        # 使用近似相等断言检查两者是否接近
        assert_almost_equal(gamln, lngam, 8)

    # 测试特殊函数 gammainccinv 的正确性
    def test_gammainccinv(self):
        # 计算 gamma 不完全互补函数的逆
        gccinv = special.gammainccinv(.5, .5)
        # 计算 gamma 不完全函数的逆
        gcinv = special.gammaincinv(.5, .5)
        # 使用近似相等断言检查两者是否接近
        assert_almost_equal(gccinv, gcinv, 8)

    # 使用特殊函数错误处理装饰器，测试特殊函数 gammaincinv 的正确性
    @with_special_errors
    def test_gammaincinv(self):
        # 计算 gamma 不完全互补函数的逆
        y = special.gammaincinv(.4, .4)
        # 计算 gamma 不完全函数
        x = special.gammainc(.4, y)
        # 使用近似相等断言检查 x 是否接近 0.4
        assert_almost_equal(x, 0.4, 1)
        # 验证 gamma 不完全函数的值
        y = special.gammainc(10, 0.05)
        # 计算 gamma 不完全互补函数的逆
        x = special.gammaincinv(10, 2.5715803516000736e-20)
        # 使用近似相等断言检查两者是否接近
        assert_almost_equal(0.05, x, decimal=10)
        # 使用近似相等断言检查两者是否接近
        assert_almost_equal(y, 2.5715803516000736e-20, decimal=10)
        # 计算 gamma 不完全互补函数的逆
        x = special.gammaincinv(50, 8.20754777388471303050299243573393e-18)
        # 使用近似相等断言检查 x 是否接近 11.0
        assert_almost_equal(11.0, x, decimal=10)

    # 使用特殊函数错误处理装饰器，测试特殊函数的 975 号问题
    def test_975(self):
        # 回归测试 975 号问题 -- 算法中的切换点
        # 检查算法在该点、其周围和稍远处的工作情况
        pts = [0.25,
               np.nextafter(0.25, 0), 0.25 - 1e-12,
               np.nextafter(0.25, 1), 0.25 + 1e-12]
        # 遍历测试点
        for xp in pts:
            # 计算 gamma 不完全互补函数的逆
            y = special.gammaincinv(.4, xp)
            # 计算 gamma 不完全函数
            x = special.gammainc(0.4, y)
            # 使用近似全部相等断言检查 x 是否接近 xp
            assert_allclose(x, xp, rtol=1e-12)

    # 测试特殊函数 rgamma 的正确性
    def test_rgamma(self):
        # 计算 gamma 函数的倒数
        rgam = special.rgamma(8)
        # 计算 gamma 函数的倒数
        rlgam = 1 / special.gamma(8)
        # 使用近似相等断言检查两者是否接近
        assert_almost_equal(rgam, rlgam, 8)

    # 测试特殊函数 gamma 在负整数处的性质
    def test_infinity(self):
        # 验证 gamma 函数在负整数处是否返回无穷大
        assert_(np.isinf(special.gamma(-1)))
        # 验证 rgamma 函数在负整数处是否返回 0
        assert_equal(special.rgamma(-1), 0)
class TestHankel:

    # 测试特殊函数 hankel1 的负值性质
    def test_negv1(self):
        assert_almost_equal(special.hankel1(-3,2), -special.hankel1(3,2), 14)

    # 测试特殊函数 hankel1 的计算准确性
    def test_hankel1(self):
        # 计算特殊函数 hankel1 的值
        hank1 = special.hankel1(1,.1)
        # 计算参考值（通过其他特殊函数 jv 和 yv 计算）
        hankrl = (special.jv(1,.1) + special.yv(1,.1)*1j)
        # 断言 hankel1 的计算结果与参考值几乎相等
        assert_almost_equal(hank1,hankrl,8)

    # 测试特殊函数 hankel1e 的负值性质
    def test_negv1e(self):
        assert_almost_equal(special.hankel1e(-3,2), -special.hankel1e(3,2), 14)

    # 测试特殊函数 hankel1e 的计算准确性
    def test_hankel1e(self):
        # 计算特殊函数 hankel1e 的值
        hank1e = special.hankel1e(1,.1)
        # 计算参考值（通过特殊函数 hankel1 计算）
        hankrle = special.hankel1(1,.1)*exp(-.1j)
        # 断言 hankel1e 的计算结果与参考值几乎相等
        assert_almost_equal(hank1e,hankrle,8)

    # 测试特殊函数 hankel2 的负值性质
    def test_negv2(self):
        assert_almost_equal(special.hankel2(-3,2), -special.hankel2(3,2), 14)

    # 测试特殊函数 hankel2 的计算准确性
    def test_hankel2(self):
        # 计算特殊函数 hankel2 的值
        hank2 = special.hankel2(1,.1)
        # 计算参考值（通过其他特殊函数 jv 和 yv 计算）
        hankrl2 = (special.jv(1,.1) - special.yv(1,.1)*1j)
        # 断言 hankel2 的计算结果与参考值几乎相等
        assert_almost_equal(hank2,hankrl2,8)

    # 测试特殊函数 hankel2e 的负值性质
    def test_neg2e(self):
        assert_almost_equal(special.hankel2e(-3,2), -special.hankel2e(3,2), 14)

    # 测试特殊函数 hankel2e 的计算准确性
    def test_hankl2e(self):
        # 计算特殊函数 hankel2e 的值
        hank2e = special.hankel2e(1,.1)
        # 计算参考值（通过特殊函数 hankel2e 自身计算）
        hankrl2e = special.hankel2e(1,.1)
        # 断言 hankel2e 的计算结果与参考值几乎相等
        assert_almost_equal(hank2e,hankrl2e,8)


class TestHyper:

    # 测试特殊函数 h1vp 的计算准确性
    def test_h1vp(self):
        # 计算特殊函数 h1vp 的值
        h1 = special.h1vp(1,.1)
        # 计算参考值（通过特殊函数 jvp 和 yvp 计算）
        h1real = (special.jvp(1,.1) + special.yvp(1,.1)*1j)
        # 断言 h1vp 的计算结果与参考值几乎相等
        assert_almost_equal(h1,h1real,8)

    # 测试特殊函数 h2vp 的计算准确性
    def test_h2vp(self):
        # 计算特殊函数 h2vp 的值
        h2 = special.h2vp(1,.1)
        # 计算参考值（通过特殊函数 jvp 和 yvp 计算）
        h2real = (special.jvp(1,.1) - special.yvp(1,.1)*1j)
        # 断言 h2vp 的计算结果与参考值几乎相等
        assert_almost_equal(h2,h2real,8)

    # 测试特殊函数 hyp0f1 的计算准确性和功能特性
    def test_hyp0f1(self):
        # 标量输入的情况
        assert_allclose(special.hyp0f1(2.5, 0.5), 1.21482702689997, rtol=1e-12)
        assert_allclose(special.hyp0f1(2.5, 0), 1.0, rtol=1e-15)

        # 浮点数输入，期望值与 mpmath 匹配
        x = special.hyp0f1(3.0, [-1.5, -1, 0, 1, 1.5])
        expected = np.array([0.58493659229143, 0.70566805723127, 1.0,
                             1.37789689539747, 1.60373685288480])
        assert_allclose(x, expected, rtol=1e-12)

        # 复数输入
        x = special.hyp0f1(3.0, np.array([-1.5, -1, 0, 1, 1.5]) + 0.j)
        assert_allclose(x, expected.astype(complex), rtol=1e-12)

        # 测试广播功能
        x1 = [0.5, 1.5, 2.5]
        x2 = [0, 1, 0.5]
        x = special.hyp0f1(x1, x2)
        expected = [1.0, 1.8134302039235093, 1.21482702689997]
        assert_allclose(x, expected, rtol=1e-12)
        x = special.hyp0f1(np.vstack([x1] * 2), x2)
        assert_allclose(x, np.vstack([expected] * 2), rtol=1e-12)
        assert_raises(ValueError, special.hyp0f1,
                      np.vstack([x1] * 3), [0, 1])

    # 测试特殊函数 hyp0f1 的特定 bug（GH5764）修复情况
    def test_hyp0f1_gh5764(self):
        # 只检查失败的点；更全面的测试在 test_mpmath 中
        res = special.hyp0f1(0.8, 0.5 + 0.5*1J)
        # 预期值由 mpmath 生成
        assert_almost_equal(res, 1.6139719776441115 + 1J*0.80893054061790665)
    # 定义一个测试函数，用于测试 special.hyp1f1 函数在给定参数下的计算结果是否准确
    def test_hyp1f1_gh2957(self):
        # 调用 special.hyp1f1 函数计算给定参数下的超几何函数值
        hyp1 = special.hyp1f1(0.5, 1.5, -709.7827128933)
        hyp2 = special.hyp1f1(0.5, 1.5, -709.7827128934)
        # 断言两个计算结果在指定精度下近似相等
        assert_almost_equal(hyp1, hyp2, 12)

    # 定义另一个测试函数，用于测试 special.hyp1f1 在特定参数下的计算结果是否正确
    def test_hyp1f1_gh2282(self):
        # 调用 special.hyp1f1 函数计算给定参数下的超几何函数值
        hyp = special.hyp1f1(0.5, 1.5, -1000)
        # 断言计算结果与预期值在指定精度下近似相等
        assert_almost_equal(hyp, 0.028024956081989643, 12)

    # 定义测试函数，用于测试 special.hyp2f1 函数在多个特殊情况下的计算结果
    def test_hyp2f1(self):
        # 定义一个包含特殊情况的列表，每个条目包含函数参数和预期值
        values = [
            [0.5, 1, 1.5, 0.2**2, 0.5/0.2*log((1+0.2)/(1-0.2))],
            [0.5, 1, 1.5, -0.2**2, 1./0.2*arctan(0.2)],
            [1, 1, 2, 0.2, -1/0.2*log(1-0.2)],
            [3, 3.5, 1.5, 0.2**2, 0.5/0.2/(-5)*((1+0.2)**(-5)-(1-0.2)**(-5))],
            [-3, 3, 0.5, sin(0.2)**2, cos(2*3*0.2)],
            [3, 4, 8, 1,
             special.gamma(8) * special.gamma(8-4-3)
             / special.gamma(8-3) / special.gamma(8-4)],
            [3, 2, 3-2+1, -1,
             1./2**3*sqrt(pi) * special.gamma(1+3-2)
             / special.gamma(1+0.5*3-2) / special.gamma(0.5+0.5*3)],
            [5, 2, 5-2+1, -1,
             1./2**5*sqrt(pi) * special.gamma(1+5-2)
             / special.gamma(1+0.5*5-2) / special.gamma(0.5+0.5*5)],
            [4, 0.5+4, 1.5-2*4, -1./3,
             (8./9)**(-2*4)*special.gamma(4./3) * special.gamma(1.5-2*4)
             / special.gamma(3./2) / special.gamma(4./3-2*4)],
            # 还有一些其他情况
            # ticket #424
            [1.5, -0.5, 1.0, -10.0, 4.1300097765277476484],
            # 当 a 或 b 是负整数，c-a-b 是整数，且 x > 0.9 时
            [-2,3,1,0.95,0.715],
            [2,-3,1,0.95,-0.007],
            [-6,3,1,0.95,0.0000810625],
            [2,-5,1,0.95,-0.000029375],
            # 大的负整数情况
            (10, -900, 10.5, 0.99, 1.91853705796607664803709475658e-24),
            (10, -900, -10.5, 0.99, 3.54279200040355710199058559155e-18),
        ]
        # 遍历特殊情况列表，计算 special.hyp2f1 的结果并进行断言
        for i, (a, b, c, x, v) in enumerate(values):
            cv = special.hyp2f1(a, b, c, x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    # 定义测试函数，用于测试 special.hyperu 函数在特定情况下的计算结果
    def test_hyperu(self):
        # 调用 special.hyperu 函数计算给定参数下的超几何函数值
        val1 = special.hyperu(1,0.1,100)
        # 断言计算结果与预期值在指定精度下近似相等
        assert_almost_equal(val1,0.0098153,7)
        # 定义数组 a 和 b，用于进一步的计算
        a,b = [0.3,0.6,1.2,-2.7],[1.5,3.2,-0.4,-3.2]
        a,b = asarray(a), asarray(b)
        z = 0.5
        # 计算 special.hyperu 的结果
        hypu = special.hyperu(a,b,z)
        # 计算预期的结果 hprl
        hprl = (pi/sin(pi*b))*(special.hyp1f1(a,b,z) /
                               (special.gamma(1+a-b)*special.gamma(b)) -
                               z**(1-b)*special.hyp1f1(1+a-b,2-b,z)
                               / (special.gamma(a)*special.gamma(2-b)))
        # 断言计算结果与预期结果在指定精度下近似相等
        assert_array_almost_equal(hypu,hprl,12)

    # 定义另一个测试函数，用于测试特定情况下 special.hyperu 函数的计算结果
    def test_hyperu_gh2287(self):
        # 调用 special.hyperu 函数计算给定参数下的超几何函数值
        assert_almost_equal(special.hyperu(1, 1.5, 20.2),
                            0.048360918656699191, 12)
class TestBessel:
    # 测试特殊函数 special.itj0y0() 的返回值是否正确
    def test_itj0y0(self):
        # 调用 special.itj0y0() 计算参数为 0.2 时的结果并转为数组
        it0 = array(special.itj0y0(.2))
        # 使用 assert_array_almost_equal 断言 it0 的值与预期数组的值在8位精度内相等
        assert_array_almost_equal(
            it0,
            array([0.19933433254006822, -0.34570883800412566]),
            8,
        )

    # 测试特殊函数 special.it2j0y0() 的返回值是否正确
    def test_it2j0y0(self):
        # 调用 special.it2j0y0() 计算参数为 0.2 时的结果并转为数组
        it2 = array(special.it2j0y0(.2))
        # 使用 assert_array_almost_equal 断言 it2 的值与预期数组的值在8位精度内相等
        assert_array_almost_equal(
            it2,
            array([0.0049937546274601858, -0.43423067011231614]),
            8,
        )

    # 测试特殊函数 special.iv() 对于负整数阶与正整数阶的返回值是否相等
    def test_negv_iv(self):
        # 断言 special.iv(3,2) 和 special.iv(-3,2) 的值相等
        assert_equal(special.iv(3,2), special.iv(-3,2))

    # 测试特殊函数 special.j0() 的返回值是否正确
    def test_j0(self):
        # 计算 special.j0(.1) 和 special.jn(0,.1) 的值
        oz = special.j0(.1)
        ozr = special.jn(0,.1)
        # 使用 assert_almost_equal 断言 oz 和 ozr 的值在8位精度内相等
        assert_almost_equal(oz,ozr,8)

    # 测试特殊函数 special.j1() 的返回值是否正确
    def test_j1(self):
        # 计算 special.j1(.1) 和 special.jn(1,.1) 的值
        o1 = special.j1(.1)
        o1r = special.jn(1,.1)
        # 使用 assert_almost_equal 断言 o1 和 o1r 的值在8位精度内相等
        assert_almost_equal(o1,o1r,8)

    # 测试特殊函数 special.jn() 的返回值是否正确
    def test_jn(self):
        # 计算 special.jn(1,.2) 的值
        jnnr = special.jn(1,.2)
        # 使用 assert_almost_equal 断言 jnnr 的值与预期值在8位精度内相等
        assert_almost_equal(jnnr,0.099500832639235995,8)

    # 测试特殊函数 special.jv() 对于负整数阶的返回值是否正确
    def test_negv_jv(self):
        # 使用 assert_almost_equal 断言 special.jv(-3,2) 和 -special.jv(3,2) 的值在14位精度内相等
        assert_almost_equal(special.jv(-3,2), -special.jv(3,2), 14)

    # 测试特殊函数 special.jv() 的返回值是否正确
    def test_jv(self):
        # 定义测试数据列表
        values = [[0, 0.1, 0.99750156206604002],
                  [2./3, 1e-8, 0.3239028506761532e-5],
                  [2./3, 1e-10, 0.1503423854873779e-6],
                  [3.1, 1e-10, 0.1711956265409013e-32],
                  [2./3, 4.0, -0.2325440850267039],
                  ]
        # 遍历测试数据
        for i, (v, x, y) in enumerate(values):
            # 计算 special.jv(v, x) 的值
            yc = special.jv(v, x)
            # 使用 assert_almost_equal 断言 yc 的值与预期值 y 在8位精度内相等，提供错误消息中的测试序号
            assert_almost_equal(yc, y, 8, err_msg='test #%d' % i)

    # 测试特殊函数 special.jve() 对于负整数阶的返回值是否正确
    def test_negv_jve(self):
        # 使用 assert_almost_equal 断言 special.jve(-3,2) 和 -special.jve(3,2) 的值在14位精度内相等
        assert_almost_equal(special.jve(-3,2), -special.jve(3,2), 14)

    # 测试特殊函数 special.jve() 的返回值是否正确
    def test_jve(self):
        # 计算 special.jve(1,.2) 的值
        jvexp = special.jve(1,.2)
        # 使用 assert_almost_equal 断言 jvexp 的值与预期值在8位精度内相等
        assert_almost_equal(jvexp,0.099500832639235995,8)
        # 计算 special.jve(1,.2+1j) 的值和理论值
        jvexp1 = special.jve(1,.2+1j)
        z = .2+1j
        jvexpr = special.jv(1,z)*exp(-abs(z.imag))
        # 使用 assert_almost_equal 断言 jvexp1 的值与 jvexpr 的值在8位精度内相等
        assert_almost_equal(jvexp1,jvexpr,8)
    # 定义测试函数 test_jn_zeros，用于测试 special 模块中的 jn_zeros 函数
    def test_jn_zeros(self):
        # 调用 special 模块中的 jn_zeros 函数，计算第 0 阶贝塞尔函数的前五个零点
        jn0 = special.jn_zeros(0,5)
        # 调用 special 模块中的 jn_zeros 函数，计算第 1 阶贝塞尔函数的前五个零点
        jn1 = special.jn_zeros(1,5)
        # 使用 assert_array_almost_equal 断言函数，验证 jn0 的计算结果
        assert_array_almost_equal(jn0,array([2.4048255577,
                                              5.5200781103,
                                              8.6537279129,
                                              11.7915344391,
                                              14.9309177086]),4)
        # 使用 assert_array_almost_equal 断言函数，验证 jn1 的计算结果
        assert_array_almost_equal(jn1,array([3.83171,
                                              7.01559,
                                              10.17347,
                                              13.32369,
                                              16.47063]),4)

        # 调用 special 模块中的 jn_zeros 函数，计算第 102 阶贝塞尔函数的前五个零点
        jn102 = special.jn_zeros(102,5)
        # 使用 assert_allclose 断言函数，验证 jn102 的计算结果
        assert_allclose(jn102, array([110.89174935992040343,
                                       117.83464175788308398,
                                       123.70194191713507279,
                                       129.02417238949092824,
                                       134.00114761868422559]), rtol=1e-13)

        # 调用 special 模块中的 jn_zeros 函数，计算第 301 阶贝塞尔函数的前五个零点
        jn301 = special.jn_zeros(301,5)
        # 使用 assert_allclose 断言函数，验证 jn301 的计算结果
        assert_allclose(jn301, array([313.59097866698830153,
                                       323.21549776096288280,
                                       331.22338738656748796,
                                       338.39676338872084500,
                                       345.03284233056064157]), rtol=1e-13)

    # 定义测试函数 test_jn_zeros_slow，用于测试特定情况下 special 模块中的 jn_zeros 函数
    def test_jn_zeros_slow(self):
        # 调用 special 模块中的 jn_zeros 函数，计算第 0 阶贝塞尔函数的前 300 个零点
        jn0 = special.jn_zeros(0, 300)
        # 使用 assert_allclose 断言函数，验证 jn0 中特定索引处的计算结果
        assert_allclose(jn0[260-1], 816.02884495068867280, rtol=1e-13)
        assert_allclose(jn0[280-1], 878.86068707124422606, rtol=1e-13)
        assert_allclose(jn0[300-1], 941.69253065317954064, rtol=1e-13)

        # 调用 special 模块中的 jn_zeros 函数，计算第 10 阶贝塞尔函数的前 300 个零点
        jn10 = special.jn_zeros(10, 300)
        # 使用 assert_allclose 断言函数，验证 jn10 中特定索引处的计算结果
        assert_allclose(jn10[260-1], 831.67668514305631151, rtol=1e-13)
        assert_allclose(jn10[280-1], 894.51275095371316931, rtol=1e-13)
        assert_allclose(jn10[300-1], 957.34826370866539775, rtol=1e-13)

        # 调用 special 模块中的 jn_zeros 函数，计算第 3010 阶贝塞尔函数的前五个零点
        jn3010 = special.jn_zeros(3010,5)
        # 使用 assert_allclose 断言函数，验证 jn3010 的计算结果
        assert_allclose(jn3010, array([3036.86590780927,
                                        3057.06598526482,
                                        3073.66360690272,
                                        3088.37736494778,
                                        3101.86438139042]), rtol=1e-8)

    # 定义测试函数 test_jnjnp_zeros，用于测试 special 模块中的 jnjnp_zeros 函数
    def test_jnjnp_zeros(self):
        # 将 special 模块中的 jn 函数赋值给变量 jn
        jn = special.jn

        # 定义 jnp 函数，计算贝塞尔函数的导数
        def jnp(n, x):
            return (jn(n-1,x) - jn(n+1,x))/2
        
        # 遍历特定范围内的整数，调用 special 模块中的 jnjnp_zeros 函数
        for nt in range(1, 30):
            # 解包 jnjnp_zeros 函数的返回值，获取贝塞尔函数的零点和相关信息
            z, n, m, t = special.jnjnp_zeros(nt)
            # 遍历 z, n, t 中的元素，分别进行断言检查
            for zz, nn, tt in zip(z, n, t):
                if tt == 0:
                    # 使用 assert_allclose 断言函数，验证对应零点处的贝塞尔函数值接近于 0
                    assert_allclose(jn(nn, zz), 0, atol=1e-6)
                elif tt == 1:
                    # 使用 assert_allclose 断言函数，验证对应零点处的贝塞尔函数导数值接近于 0
                    assert_allclose(jnp(nn, zz), 0, atol=1e-6)
                else:
                    # 抛出 AssertionError，如果返回的 t 值不是 0 或 1
                    raise AssertionError("Invalid t return for nt=%d" % nt)
    # 定义测试函数 test_jnp_zeros，用于测试 special 模块中 jnp_zeros 函数的行为
    def test_jnp_zeros(self):
        # 调用 jnp_zeros 函数生成一组特殊函数的零点，存储在 jnp 变量中
        jnp = special.jnp_zeros(1,5)
        # 断言 jnp 变量近似等于给定的数值数组，精确到小数点后四位
        assert_array_almost_equal(jnp, array([1.84118,
                                              5.33144,
                                              8.53632,
                                              11.70600,
                                              14.86359]), 4)
        # 再次调用 jnp_zeros 函数生成另一组特殊函数的零点，存储在 jnp 变量中
        jnp = special.jnp_zeros(443,5)
        # 断言特殊函数 jvp(443, jnp) 的返回值近似等于 0，允许的误差为 1e-15
        assert_allclose(special.jvp(443, jnp), 0, atol=1e-15)

    # 定义测试函数 test_jnyn_zeros，用于测试 special 模块中 jnyn_zeros 函数的行为
    def test_jnyn_zeros(self):
        # 调用 jnyn_zeros 函数生成一组特殊函数的零点，存储在 jnz 变量中
        jnz = special.jnyn_zeros(1,5)
        # 断言 jnz 变量近似等于给定的数值元组，包含多个数组，精确到小数点后五位
        assert_array_almost_equal(jnz, (array([3.83171,
                                               7.01559,
                                               10.17347,
                                               13.32369,
                                               16.47063]),
                                       array([1.84118,
                                               5.33144,
                                               8.53632,
                                               11.70600,
                                               14.86359]),
                                       array([2.19714,
                                               5.42968,
                                               8.59601,
                                               11.74915,
                                               14.89744]),
                                       array([3.68302,
                                               6.94150,
                                               10.12340,
                                               13.28576,
                                               16.44006])), 5)

    # 定义测试函数 test_jvp，用于测试 special 模块中 jvp 函数的行为
    def test_jvp(self):
        # 调用 jvp 函数计算特殊函数 jvp(2,2)，存储在 jvprim 变量中
        jvprim = special.jvp(2,2)
        # 计算特殊函数 jv(1,2) 和 jv(3,2) 的差的一半，存储在 jv0 变量中
        jv0 = (special.jv(1,2)-special.jv(3,2))/2
        # 断言 jvprim 变量近似等于 jv0 变量，精确到小数点后十位
        assert_almost_equal(jvprim, jv0, 10)

    # 定义测试函数 test_k0，用于测试 special 模块中 k0 函数的行为
    def test_k0(self):
        # 调用 k0 函数计算特殊函数 k0(.1)，存储在 ozk 变量中
        ozk = special.k0(.1)
        # 调用 kv 函数计算特殊函数 kv(0, .1)，存储在 ozkr 变量中
        ozkr = special.kv(0, .1)
        # 断言 ozk 变量近似等于 ozkr 变量，精确到小数点后八位
        assert_almost_equal(ozk, ozkr, 8)

    # 定义测试函数 test_k0e，用于测试 special 模块中 k0e 函数的行为
    def test_k0e(self):
        # 调用 k0e 函数计算特殊函数 k0e(.1)，存储在 ozke 变量中
        ozke = special.k0e(.1)
        # 调用 kve 函数计算特殊函数 kve(0, .1)，存储在 ozker 变量中
        ozker = special.kve(0, .1)
        # 断言 ozke 变量近似等于 ozker 变量，精确到小数点后八位
        assert_almost_equal(ozke, ozker, 8)

    # 定义测试函数 test_k1，用于测试 special 模块中 k1 函数的行为
    def test_k1(self):
        # 调用 k1 函数计算特殊函数 k1(.1)，存储在 o1k 变量中
        o1k = special.k1(.1)
        # 调用 kv 函数计算特殊函数 kv(1, .1)，存储在 o1kr 变量中
        o1kr = special.kv(1, .1)
        # 断言 o1k 变量近似等于 o1kr 变量，精确到小数点后八位
        assert_almost_equal(o1k, o1kr, 8)

    # 定义测试函数 test_k1e，用于测试 special 模块中 k1e 函数的行为
    def test_k1e(self):
        # 调用 k1e 函数计算特殊函数 k1e(.1)，存储在 o1ke 变量中
        o1ke = special.k1e(.1)
        # 调用 kve 函数计算特殊函数 kve(1, .1)，存储在 o1ker 变量中
        o1ker = special.kve(1, .1)
        # 断言 o1ke 变量近似等于 o1ker 变量，精确到小数点后八位
        assert_almost_equal(o1ke, o1ker, 8)
    # 测试 Jacobi 多项式的计算
    def test_jacobi(self):
        # 生成随机的参数 a 和 b
        a = 5*np.random.random() - 1
        b = 5*np.random.random() - 1
        # 计算 Jacobi 多项式 P0、P1、P2 和 P3
        P0 = special.jacobi(0,a,b)
        P1 = special.jacobi(1,a,b)
        P2 = special.jacobi(2,a,b)
        P3 = special.jacobi(3,a,b)

        # 断言 P0 的系数 c 等于 [1]
        assert_array_almost_equal(P0.c,[1],13)
        # 断言 P1 的系数 c 等于 [a+b+2, (a-b)/2.0]
        assert_array_almost_equal(P1.c,array([a+b+2,a-b])/2.0,13)
        
        # 计算 P2 的系数 cp，并通过数学计算转换为 p2c
        cp = [(a+b+3)*(a+b+4), 4*(a+b+3)*(a+2), 4*(a+1)*(a+2)]
        p2c = [cp[0],cp[1]-2*cp[0],cp[2]-cp[1]+cp[0]]
        # 断言 P2 的系数 c 等于 p2c 的每个元素除以 8.0
        assert_array_almost_equal(P2.c,array(p2c)/8.0,13)
        
        # 计算 P3 的系数 cp，并通过数学计算转换为 p3c
        cp = [(a+b+4)*(a+b+5)*(a+b+6),6*(a+b+4)*(a+b+5)*(a+3),
              12*(a+b+4)*(a+2)*(a+3),8*(a+1)*(a+2)*(a+3)]
        p3c = [cp[0],cp[1]-3*cp[0],cp[2]-2*cp[1]+3*cp[0],cp[3]-cp[2]+cp[1]-cp[0]]
        # 断言 P3 的系数 c 等于 p3c 的每个元素除以 48.0
        assert_array_almost_equal(P3.c,array(p3c)/48.0,13)

    # 测试 modified Bessel 函数 kn 的计算
    def test_kn(self):
        # 计算 kn1
        kn1 = special.kn(0,.2)
        # 断言 kn1 的值近似等于 1.7527038555281462，精度为 8
        assert_almost_equal(kn1,1.7527038555281462,8)

    # 测试定理函数 kv 的计算
    def test_negv_kv(self):
        # 断言 kv(3.0, 2.2) 等于 kv(-3.0, 2.2)
        assert_equal(special.kv(3.0, 2.2), special.kv(-3.0, 2.2))

    # 测试 kv 函数在 v=0 时的计算
    def test_kv0(self):
        # 计算 kv0
        kv0 = special.kv(0,.2)
        # 断言 kv0 的值近似等于 1.7527038555281462，精度为 10
        assert_almost_equal(kv0, 1.7527038555281462, 10)

    # 测试 kv 函数在 v=1 时的计算
    def test_kv1(self):
        # 计算 kv1
        kv1 = special.kv(1,0.2)
        # 断言 kv1 的值近似等于 4.775972543220472，精度为 10
        assert_almost_equal(kv1, 4.775972543220472, 10)

    # 测试 kv 函数在 v=2 时的计算
    def test_kv2(self):
        # 计算 kv2
        kv2 = special.kv(2,0.2)
        # 断言 kv2 的值近似等于 49.51242928773287，精度为 10
        assert_almost_equal(kv2, 49.51242928773287, 10)

    # 测试 modified Bessel 函数 kn 在大阶数下的计算
    def test_kn_largeorder(self):
        # 断言 kn(32, 1) 的值接近于 1.7516596664574289e+43
        assert_allclose(special.kn(32, 1), 1.7516596664574289e+43)

    # 测试 kv 函数在大参数下的计算
    def test_kv_largearg(self):
        # 断言 kv(0, 1e19) 的值等于 0
        assert_equal(special.kv(0, 1e19), 0)

    # 测试 Kelvin 函数 kve 的计算
    def test_negv_kve(self):
        # 断言 kve(3.0, 2.2) 等于 kve(-3.0, 2.2)
        assert_equal(special.kve(3.0, 2.2), special.kve(-3.0, 2.2))

    # 测试 Kelvin 函数 kve 在不同参数下的计算
    def test_kve(self):
        # 计算 kve1 和相应的 kv1
        kve1 = special.kve(0,.2)
        kv1 = special.kv(0,.2)*exp(.2)
        # 断言 kve1 和 kv1 的值近似相等，精度为 8
        assert_almost_equal(kve1,kv1,8)
        
        # 计算 kve2 和相应的 kv2
        z = .2+1j
        kve2 = special.kve(0,z)
        kv2 = special.kv(0,z)*exp(z)
        # 断言 kve2 和 kv2 的值近似相等，精度为 8
        assert_almost_equal(kve2,kv2,8)

    # 测试 modified Bessel 函数的导数函数 kvp 的计算
    def test_kvp_v0n1(self):
        z = 2.2
        # 断言 -kv(1,z) 等于 kvp(0,z, n=1)
        assert_almost_equal(-special.kv(1,z), special.kvp(0,z, n=1), 10)

    # 测试 modified Bessel 函数的导数函数 kvp 在 n=1 时的计算
    def test_kvp_n1(self):
        v = 3.
        z = 2.2
        # 计算 xc 和 x
        xc = -special.kv(v+1,z) + v/z*special.kv(v,z)
        x = special.kvp(v,z, n=1)
        # 断言 xc 等于 x，精度为 10
        assert_almost_equal(xc, x, 10)   # this function (kvp) is broken

    # 测试 modified Bessel 函数的导数函数 kvp 在 n=2 时的计算
    def test_kvp_n2(self):
        v = 3.
        z = 2.2
        # 计算 xc 和 x
        xc = (z**2+v**2-v)/z**2 * special.kv(v,z) + special.kv(v+1,z)/z
        x = special.kvp(v, z, n=2)
        # 断言 xc 等于 x，精度为 10
        assert_almost_equal(xc, x, 10)

    # 测试第一类零阶贝塞尔函数 y0 的计算
    def test_y0(self):
        # 计算 oz 和 ozr
        oz = special.y0(.1)
        ozr = special.yn(0,.1)
        # 断言 oz 和 ozr 的值近似相等，精度为 8
        assert_almost_equal(oz,ozr,8)

    # 测试第一类一阶贝塞尔函数 y1 的计算
    def test_y1(self):
        # 计算 o1 和 o1r
        o1 = special.y1(.1)
        o1r = special.yn(1,.1)
        # 断言 o1 和 o1r 的值近似相等，精度为 8
        assert_almost_equal(o1,o1r,8)

    # 测试第一类零阶贝塞尔函数的零点和导数 y0_zeros 的计算
    def test_y0_zeros(self):
        # 计算 yo 和 y
    # 测试特殊函数 `y1_zeros` 的返回值是否与期望的几乎相等
    def test_y1_zeros(self):
        y1 = special.y1_zeros(1)
        assert_array_almost_equal(y1,(array([2.19714]),array([0.52079])),5)

    # 测试带有复数参数的特殊函数 `y1p_zeros` 的返回值是否几乎相等
    def test_y1p_zeros(self):
        y1p = special.y1p_zeros(1,complex=1)
        assert_array_almost_equal(
            y1p,
            (array([0.5768+0.904j]), array([-0.7635+0.5892j])),
            3,
        )

    # 测试特殊函数 `yn_zeros` 在不同参数下的返回值是否几乎相等
    def test_yn_zeros(self):
        an = special.yn_zeros(4,2)
        assert_array_almost_equal(an,array([5.64515, 9.36162]),5)
        an = special.yn_zeros(443,5)
        assert_allclose(an, [450.13573091578090314,
                             463.05692376675001542,
                             472.80651546418663566,
                             481.27353184725625838,
                             488.98055964441374646],
                        rtol=1e-15,)

    # 测试特殊函数 `ynp_zeros` 的返回值是否几乎为零
    def test_ynp_zeros(self):
        ao = special.ynp_zeros(0,2)
        assert_array_almost_equal(ao,array([2.19714133, 5.42968104]),6)
        ao = special.ynp_zeros(43,5)
        assert_allclose(special.yvp(43, ao), 0, atol=1e-15)
        ao = special.ynp_zeros(443,5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-9)

    # 测试特殊函数 `ynp_zeros` 对于大阶数的返回值是否几乎为零
    def test_ynp_zeros_large_order(self):
        ao = special.ynp_zeros(443,5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-14)

    # 测试特殊函数 `yn` 的返回值是否准确
    def test_yn(self):
        yn2n = special.yn(1,.2)
        assert_almost_equal(yn2n,-3.3238249881118471,8)

    # 测试特殊函数 `yn` 在边界条件下的返回值是否正确
    def test_yn_gh_20405(self):
        # 强制检查大阶数时的正确渐近行为
        observed = cephes.yn(500, 1)
        assert observed == -np.inf

    # 测试特殊函数 `yv` 的返回值是否准确
    def test_negv_yv(self):
        assert_almost_equal(special.yv(-3,2), -special.yv(3,2), 14)

    # 测试特殊函数 `yv` 的返回值是否准确
    def test_yv(self):
        yv2 = special.yv(1,.2)
        assert_almost_equal(yv2,-3.3238249881118471,8)

    # 测试特殊函数 `yve` 的返回值是否准确
    def test_negv_yve(self):
        assert_almost_equal(special.yve(-3,2), -special.yve(3,2), 14)

    # 测试特殊函数 `yve` 的返回值是否准确
    def test_yve(self):
        yve2 = special.yve(1,.2)
        assert_almost_equal(yve2,-3.3238249881118471,8)
        yve2r = special.yv(1,.2+1j)*exp(-1)
        yve22 = special.yve(1,.2+1j)
        assert_almost_equal(yve22,yve2r,8)

    # 测试特殊函数 `yvp` 的返回值是否与期望的几乎相等
    def test_yvp(self):
        yvpr = (special.yv(1,.2) - special.yv(3,.2))/2.0
        yvp1 = special.yvp(2,.2)
        assert_array_almost_equal(yvp1,yvpr,10)

    # 准备进行 Cephes 实现与 AMOS 比较的点的生成器函数
    def _cephes_vs_amos_points(self):
        """Yield points at which to compare Cephes implementation to AMOS"""
        # 检查多个点，包括大幅度的点
        v = [-120, -100.3, -20., -10., -1., -.5, 0., 1., 12.49, 120., 301]
        z = [-1300, -11, -10, -1, 1., 10., 200.5, 401., 600.5, 700.6, 1300,
             10003]
        yield from itertools.product(v, z)

        # 检查半整数；这些对于 cephes/iv 至少是有问题的点
        yield from itertools.product(0.5 + arange(-60, 60), [3.5])
    # 对比 Cephes 和 AMOS 函数的输出结果是否一致
    def check_cephes_vs_amos(self, f1, f2, rtol=1e-11, atol=0, skip=None):
        # 遍历 Cephes vs. AMOS 测试点
        for v, z in self._cephes_vs_amos_points():
            # 如果定义了跳过函数，并且满足跳过条件，则跳过当前点
            if skip is not None and skip(v, z):
                continue
            # 调用两种不同实现的函数 f1 和 f2，分别计算结果
            c1, c2, c3 = f1(v, z), f1(v,z+0j), f2(int(v), z)
            # 如果 c1 是无穷大
            if np.isinf(c1):
                # 断言 c2 的绝对值大于等于 1e300，即 c2 应该是大数
                assert_(np.abs(c2) >= 1e300, (v, z))
            # 如果 c1 是 NaN
            elif np.isnan(c1):
                # 断言 c2 的虚部不为零，即 c2 应该是一个复数
                assert_(c2.imag != 0, (v, z))
            # 否则
            else:
                # 断言 c1 和 c2 的值在相对和绝对误差范围内接近
                assert_allclose(c1, c2, err_msg=(v, z), rtol=rtol, atol=atol)
                # 如果 v 是整数
                if v == int(v):
                    # 再次断言 c3 和 c2 的值在误差范围内接近
                    assert_allclose(c3, c2, err_msg=(v, z),
                                     rtol=rtol, atol=atol)

    # 标记为预期失败的测试，如果运行在 'ppc64le' 平台上会失败
    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_jv_cephes_vs_amos(self):
        # 对 special.jv 和 special.jn 函数进行 Cephes vs. AMOS 测试
        self.check_cephes_vs_amos(special.jv, special.jn, rtol=1e-10, atol=1e-305)

    # 标记为预期失败的测试，如果运行在 'ppc64le' 平台上会失败
    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_yv_cephes_vs_amos(self):
        # 对 special.yv 和 special.yn 函数进行 Cephes vs. AMOS 测试
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305)

    # 只对小阶数进行 Cephes vs. AMOS 测试的测试用例
    def test_yv_cephes_vs_amos_only_small_orders(self):
        # 定义一个跳过函数，用于跳过大阶数的测试点
        def skipper(v, z):
            return abs(v) > 50
        # 对 special.yv 和 special.yn 函数进行 Cephes vs. AMOS 测试，但跳过大阶数的测试点
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305,
                                  skip=skipper)

    # 对 special.iv 函数进行 Cephes vs. AMOS 测试，忽略所有错误
    def test_iv_cephes_vs_amos(self):
        with np.errstate(all='ignore'):
            self.check_cephes_vs_amos(special.iv, special.iv, rtol=5e-9, atol=1e-305)

    # 标记为慢速测试的测试用例，对特定参数下的大规模 Cephes vs. AMOS 测试
    @pytest.mark.slow
    def test_iv_cephes_vs_amos_mass_test(self):
        N = 1000000
        np.random.seed(1)
        # 生成随机参数 v 和 x
        v = np.random.pareto(0.5, N) * (-1)**np.random.randint(2, size=N)
        x = np.random.pareto(0.2, N) * (-1)**np.random.randint(2, size=N)
        # 随机选择一些点并将 v 转换为整数
        imsk = (np.random.randint(8, size=N) == 0)
        v[imsk] = v[imsk].astype(np.int64)
        # 忽略所有错误并计算 special.iv 函数的结果
        with np.errstate(all='ignore'):
            c1 = special.iv(v, x)
            c2 = special.iv(v, x+0j)
            # 处理无穷大和零的差异
            c1[abs(c1) > 1e300] = np.inf
            c2[abs(c2) > 1e300] = np.inf
            c1[abs(c1) < 1e-300] = 0
            c2[abs(c2) < 1e-300] = 0
            # 计算相对误差
            dc = abs(c1/c2 - 1)
            dc[np.isnan(dc)] = 0
        # 找到最大的相对误差点
        k = np.argmax(dc)
        # 断言在最大相对误差点处的误差小于 2e-7
        assert_(
            dc[k] < 2e-7,
            (v[k], x[k], special.iv(v[k], x[k]), special.iv(v[k], x[k]+0j))
        )

    # 对 special.kv 和 special.kn 函数进行 Cephes vs. AMOS 测试
    def test_kv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.kv, special.kn, rtol=1e-9, atol=1e-305)
        self.check_cephes_vs_amos(special.kv, special.kv, rtol=1e-9, atol=1e-305)
    def test_ticket_623(self):
        # 测试特殊函数 special.jv 的返回值是否接近给定值
        assert_allclose(special.jv(3, 4), 0.43017147387562193)
        assert_allclose(special.jv(301, 1300), 0.0183487151115275)
        assert_allclose(special.jv(301, 1296.0682), -0.0224174325312048)

    def test_ticket_853(self):
        """Negative-order Bessels"""
        # 对负阶贝塞尔函数进行测试

        # cephes 库中的特殊函数
        assert_allclose(special.jv(-1, 1), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1), -1.650682606816255)
        assert_allclose(special.iv(-1, 1), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1), 1.624838898635178)
        assert_allclose(special.jv(-0.5, 1), 0.43109886801837607952)
        assert_allclose(special.yv(-0.5, 1), 0.6713967071418031)
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)
        assert_allclose(special.kv(-0.5, 1), 0.4610685044478945)

        # amos 库中的特殊函数
        assert_allclose(special.jv(-1, 1+0j), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1+0j), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1+0j), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1+0j), -1.650682606816255)
        assert_allclose(special.iv(-1, 1+0j), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1+0j), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1+0j), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1+0j), 1.624838898635178)
        assert_allclose(special.jv(-0.5, 1+0j), 0.43109886801837607952)
        assert_allclose(special.jv(-0.5, 1+1j), 0.2628946385649065-0.827050182040562j)
        assert_allclose(special.yv(-0.5, 1+0j), 0.6713967071418031)
        assert_allclose(special.yv(-0.5, 1+1j), 0.967901282890131+0.0602046062142816j)
        assert_allclose(special.iv(-0.5, 1+0j), 1.231200214592967)
        assert_allclose(special.iv(-0.5, 1+1j), 0.77070737376928+0.39891821043561j)
        assert_allclose(special.kv(-0.5, 1+0j), 0.4610685044478945)
        assert_allclose(special.kv(-0.5, 1+1j), 0.06868578341999-0.38157825981268j)

        # 通过数学关系验证特殊函数的一些性质
        assert_allclose(special.jve(-0.5,1+0.3j), special.jv(-0.5, 1+0.3j)*exp(-0.3))
        assert_allclose(special.yve(-0.5,1+0.3j), special.yv(-0.5, 1+0.3j)*exp(-0.3))
        assert_allclose(special.ive(-0.5,0.3+1j), special.iv(-0.5, 0.3+1j)*exp(-0.3))
        assert_allclose(special.kve(-0.5,0.3+1j), special.kv(-0.5, 0.3+1j)*exp(0.3+1j))

        # 鱼类函数的汉克尔变换的验证
        assert_allclose(
            special.hankel1(-0.5, 1+1j),
            special.jv(-0.5, 1+1j) + 1j*special.yv(-0.5,1+1j)
        )
        assert_allclose(
            special.hankel2(-0.5, 1+1j),
            special.jv(-0.5, 1+1j) - 1j*special.yv(-0.5,1+1j)
        )
    def test_ticket_854(self):
        """Real-valued Bessel domains"""
        # 检查特殊函数在给定参数下是否返回 NaN
        assert_(isnan(special.jv(0.5, -1)))
        assert_(isnan(special.iv(0.5, -1)))
        assert_(isnan(special.yv(0.5, -1)))
        assert_(isnan(special.yv(1, -1)))
        assert_(isnan(special.kv(0.5, -1)))
        assert_(isnan(special.kv(1, -1)))
        assert_(isnan(special.jve(0.5, -1)))
        assert_(isnan(special.ive(0.5, -1)))
        assert_(isnan(special.yve(0.5, -1)))
        assert_(isnan(special.yve(1, -1)))
        assert_(isnan(special.kve(0.5, -1)))
        assert_(isnan(special.kve(1, -1)))
        # 检查 Airy 函数在参数为 -1 时的返回值是否包含 NaN，前两个元素
        assert_(isnan(special.airye(-1)[0:2]).all(), special.airye(-1))
        # 检查 Airy 函数在参数为 -1 时的返回值是否不含 NaN，后两个元素
        assert_(not isnan(special.airye(-1)[2:4]).any(), special.airye(-1))

    def test_gh_7909(self):
        # 检查特殊函数在给定参数下是否返回无穷大
        assert_(special.kv(1.5, 0) == np.inf)
        assert_(special.kve(1.5, 0) == np.inf)

    def test_ticket_503(self):
        """Real-valued Bessel I overflow"""
        # 检查修正的贝塞尔函数在给定参数下的计算值是否与预期值在一定精度内相等
        assert_allclose(special.iv(1, 700), 1.528500390233901e302)
        assert_allclose(special.iv(1000, 1120), 1.301564549405821e301)

    def test_iv_hyperg_poles(self):
        # 检查修正的贝塞尔函数在给定参数下的计算值是否与预期值在一定精度内相等
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)

    def iv_series(self, v, z, n=200):
        # 计算修正的贝塞尔函数的级数展开
        k = arange(0, n).astype(double)
        r = (v+2*k)*log(.5*z) - special.gammaln(k+1) - special.gammaln(v+k+1)
        r[isnan(r)] = inf
        r = exp(r)
        err = abs(r).max() * finfo(double).eps * n + abs(r[-1])*10
        return r.sum(), err

    def test_i0_series(self):
        # 检查修正的零阶修正贝塞尔函数的级数展开是否与标准函数值在误差限内相等
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(0, z)
            assert_allclose(special.i0(z), value, atol=err, err_msg=z)

    def test_i1_series(self):
        # 检查修正的一阶修正贝塞尔函数的级数展开是否与标准函数值在误差限内相等
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(1, z)
            assert_allclose(special.i1(z), value, atol=err, err_msg=z)

    def test_iv_series(self):
        # 检查修正的贝塞尔函数的级数展开是否与标准函数值在误差限内相等，对不同的参数组合进行测试
        for v in [-20., -10., -1., 0., 1., 12.49, 120.]:
            for z in [1., 10., 200.5, -1+2j]:
                value, err = self.iv_series(v, z)
                assert_allclose(special.iv(v, z), value, atol=err, err_msg=(v, z))

    def test_i0(self):
        # 检查零阶修正贝塞尔函数的计算值是否与预期值在指定精度内相等
        values = [[0.0, 1.0],
                  [1e-10, 1.0],
                  [0.1, 0.9071009258],
                  [0.5, 0.6450352706],
                  [1.0, 0.4657596077],
                  [2.5, 0.2700464416],
                  [5.0, 0.1835408126],
                  [20.0, 0.0897803119],
                  ]
        for i, (x, v) in enumerate(values):
            cv = special.i0(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i0e(self):
        # 检查修正的零阶修正贝塞尔函数的指数形式计算值是否与直接计算值在指定精度内相等
        oize = special.i0e(.1)
        oizer = special.ive(0,.1)
        assert_almost_equal(oize,oizer,8)
    # 定义一个测试方法 `test_i1`，用于测试特殊函数 `special.i1` 的计算结果
    def test_i1(self):
        # 定义测试用例的输入和预期输出值
        values = [[0.0, 0.0],
                  [1e-10, 0.4999999999500000e-10],
                  [0.1, 0.0452984468],
                  [0.5, 0.1564208032],
                  [1.0, 0.2079104154],
                  [5.0, 0.1639722669],
                  [20.0, 0.0875062222],
                  ]
        # 遍历每个测试用例，并使用 `enumerate` 获取索引和元组 (x, v)
        for i, (x, v) in enumerate(values):
            # 计算特殊函数 `i1(x) * exp(-x)` 的值
            cv = special.i1(x) * exp(-x)
            # 使用 `assert_almost_equal` 断言计算结果 `cv` 与预期值 `v` 相近，精度为 8
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    # 定义一个测试方法 `test_i1e`，用于测试特殊函数 `special.i1e` 的计算结果
    def test_i1e(self):
        # 计算特殊函数 `i1e(.1)` 和 `ive(1, .1)` 的值，并使用 `assert_almost_equal` 断言它们相等，精度为 8
        oi1e = special.i1e(.1)
        oi1er = special.ive(1, .1)
        assert_almost_equal(oi1e, oi1er, 8)

    # 定义一个测试方法 `test_iti0k0`，用于测试特殊函数 `special.iti0k0` 的计算结果
    def test_iti0k0(self):
        # 计算 `special.iti0k0(5)` 的结果并转换为数组 `iti0`
        iti0 = array(special.iti0k0(5))
        # 使用 `assert_array_almost_equal` 断言数组 `iti0` 与给定数组接近，精度为 5
        assert_array_almost_equal(
            iti0,
            array([31.848667776169801, 1.5673873907283657]),
            5,
        )

    # 定义一个测试方法 `test_it2i0k0`，用于测试特殊函数 `special.it2i0k0` 的计算结果
    def test_it2i0k0(self):
        # 计算 `special.it2i0k0(.1)` 的结果并转换为数组 `it2k`
        it2k = special.it2i0k0(.1)
        # 使用 `assert_array_almost_equal` 断言数组 `it2k` 与给定数组接近，精度为 6
        assert_array_almost_equal(
            it2k,
            array([0.0012503906973464409, 3.3309450354686687]),
            6,
        )

    # 定义一个测试方法 `test_iv`，用于测试特殊函数 `special.iv` 的计算结果
    def test_iv(self):
        # 计算 `special.iv(0, .1) * exp(-.1)` 的值，并使用 `assert_almost_equal` 断言其值与预期值相近，精度为 10
        iv1 = special.iv(0, .1) * exp(-.1)
        assert_almost_equal(iv1, 0.90710092578230106, 10)

    # 定义一个测试方法 `test_negv_ive`，用于测试特殊函数 `special.ive` 的计算结果
    def test_negv_ive(self):
        # 使用 `assert_equal` 断言 `special.ive(3, 2)` 与 `special.ive(-3, 2)` 的值相等
        assert_equal(special.ive(3, 2), special.ive(-3, 2))

    # 定义一个测试方法 `test_ive`，用于测试特殊函数 `special.ive` 的计算结果
    def test_ive(self):
        # 计算 `special.ive(0, .1)` 和 `special.iv(0, .1) * exp(-.1)` 的值，并使用 `assert_almost_equal` 断言它们相等，精度为 10
        ive1 = special.ive(0, .1)
        iv1 = special.iv(0, .1) * exp(-.1)
        assert_almost_equal(ive1, iv1, 10)

    # 定义一个测试方法 `test_ivp0`，用于测试特殊函数 `special.ivp` 的计算结果
    def test_ivp0(self):
        # 使用 `assert_almost_equal` 断言 `special.iv(1, 2)` 与 `special.ivp(0, 2)` 的值相近，精度为 10
        assert_almost_equal(special.iv(1, 2), special.ivp(0, 2), 10)

    # 定义一个测试方法 `test_ivp`，用于测试特殊函数 `special.ivp` 的计算结果
    def test_ivp(self):
        # 计算 `special.iv(0, 2)` 和 `special.iv(2, 2)` 的平均值，并用 `y` 表示
        y = (special.iv(0, 2) + special.iv(2, 2)) / 2
        # 计算 `special.ivp(1, 2)` 的值，并使用 `assert_almost_equal` 断言其与 `y` 的值相近，精度为 10
        x = special.ivp(1, 2)
        assert_almost_equal(x, y, 10)
# Laguerre 多项式的单元测试类
class TestLaguerre:
    # 测试 Laguerre 多项式的前几个
    def test_laguerre(self):
        # 计算 Laguerre 多项式 L_0(x)
        lag0 = special.laguerre(0)
        # 计算 Laguerre 多项式 L_1(x)
        lag1 = special.laguerre(1)
        # 计算 Laguerre 多项式 L_2(x)
        lag2 = special.laguerre(2)
        # 计算 Laguerre 多项式 L_3(x)
        lag3 = special.laguerre(3)
        # 计算 Laguerre 多项式 L_4(x)
        lag4 = special.laguerre(4)
        # 计算 Laguerre 多项式 L_5(x)
        lag5 = special.laguerre(5)
        # 断言 L_0(x) 的系数数组
        assert_array_almost_equal(lag0.c,[1],13)
        # 断言 L_1(x) 的系数数组
        assert_array_almost_equal(lag1.c,[-1,1],13)
        # 断言 L_2(x) 的系数数组
        assert_array_almost_equal(lag2.c,array([1,-4,2])/2.0,13)
        # 断言 L_3(x) 的系数数组
        assert_array_almost_equal(lag3.c,array([-1,9,-18,6])/6.0,13)
        # 断言 L_4(x) 的系数数组
        assert_array_almost_equal(lag4.c,array([1,-16,72,-96,24])/24.0,13)
        # 断言 L_5(x) 的系数数组
        assert_array_almost_equal(lag5.c,array([-1,25,-200,600,-600,120])/120.0,13)

    # 测试通用 Laguerre 多项式
    def test_genlaguerre(self):
        # 随机生成参数 k
        k = 5*np.random.random() - 0.9
        # 计算通用 Laguerre 多项式 L_0^(k)(x)
        lag0 = special.genlaguerre(0,k)
        # 计算通用 Laguerre 多项式 L_1^(k)(x)
        lag1 = special.genlaguerre(1,k)
        # 计算通用 Laguerre 多项式 L_2^(k)(x)
        lag2 = special.genlaguerre(2,k)
        # 计算通用 Laguerre 多项式 L_3^(k)(x)
        lag3 = special.genlaguerre(3,k)
        # 断言 L_0^(k)(x) 的系数数组
        assert_equal(lag0.c, [1])
        # 断言 L_1^(k)(x) 的系数数组
        assert_equal(lag1.c, [-1, k + 1])
        # 断言 L_2^(k)(x) 的系数数组
        assert_almost_equal(
            lag2.c,
            array([1,-2*(k+2),(k+1.)*(k+2.)])/2.0
        )
        # 断言 L_3^(k)(x) 的系数数组
        assert_almost_equal(
            lag3.c,
            array([-1,3*(k+3),-3*(k+2)*(k+3),(k+1)*(k+2)*(k+3)])/6.0
        )


# Legendre 多项式的单元测试类，参考 Abrahmowitz 和 Stegan
class TestLegendre:
    # 测试 Legendre 多项式的前几个
    def test_legendre(self):
        # 计算 Legendre 多项式 P_0(x)
        leg0 = special.legendre(0)
        # 计算 Legendre 多项式 P_1(x)
        leg1 = special.legendre(1)
        # 计算 Legendre 多项式 P_2(x)
        leg2 = special.legendre(2)
        # 计算 Legendre 多项式 P_3(x)
        leg3 = special.legendre(3)
        # 计算 Legendre 多项式 P_4(x)
        leg4 = special.legendre(4)
        # 计算 Legendre 多项式 P_5(x)
        leg5 = special.legendre(5)
        # 断言 P_0(x) 的系数数组
        assert_equal(leg0.c, [1])
        # 断言 P_1(x) 的系数数组
        assert_equal(leg1.c, [1,0])
        # 断言 P_2(x) 的系数数组
        assert_almost_equal(leg2.c, array([3,0,-1])/2.0, decimal=13)
        # 断言 P_3(x) 的系数数组
        assert_almost_equal(leg3.c, array([5,0,-3,0])/2.0)
        # 断言 P_4(x) 的系数数组
        assert_almost_equal(leg4.c, array([35,0,-30,0,3])/8.0)
        # 断言 P_5(x) 的系数数组
        assert_almost_equal(leg5.c, array([63,0,-70,0,15,0])/8.0)

    # 参数化测试函数，比较 lpn(n, z) 和 clpmn(0, n, z) 的结果
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('zr', [0.5241717, 12.80232, -9.699001,
                                    0.5122437, 0.1714377])
    @pytest.mark.parametrize('zi', [9.766818, 0.2999083, 8.24726, -22.84843,
                                    -0.8792666])
    def test_lpn_against_clpmn(self, n, zr, zi):
        # 计算 lpn(n, z)
        reslpn = special.lpn(n, zr + zi*1j)
        # 计算 clpmn(0, n, z)
        resclpmn = special.clpmn(0, n, zr+zi*1j)
        # 断言 lpn(n, z) 的第一个返回值与 clpmn(0, n, z) 的第一个返回值的近似程度
        assert_allclose(reslpn[0], resclpmn[0][0])
        # 断言 lpn(n, z) 的第二个返回值与 clpmn(0, n, z) 的第二个返回值的近似程度
        assert_allclose(reslpn[1], resclpmn[1][0])


# Lambda 函数的单元测试类
class TestLambda:
    # 测试 Lambda 函数
    def test_lmbda(self):
        # 计算 Lambda 函数的值
        lam = special.lmbda(1,.1)
        # 参考值数组
        lamr = (
            array([special.jn(0,.1), 2*special.jn(1,.1)/.1]),
            array([special.jvp(0,.1), -2*special.jv(1,.1)/.01 + 2*special.jvp(1,.1)/.1])
        )
        # 断言 Lambda 函数计算结果与参考值的近似程度
        assert_array_almost_equal(lam,lamr,8)


# Log1p 函数的单元测试类
class TestLog1p:
    # 测试 Log1p 函数
    def test_log1p(self):
        # 计算 Log1p 函数在不同输入上的结果
        l1p = (special.log1p(10), special.log1p(11), special.log1p(12))
        # 对数参考值数组
        l1prl = (log(11), log(12), log(13))
        # 断言 Log1p 函数计算结果与对数参考值的近似程度
        assert_array_almost_equal(l1p,l1prl,8)
    # 定义一个测试方法，用于测试特殊数学函数 log1p 的精度
    def test_log1pmore(self):
        # 计算特殊数学函数 log1p 在不同输入下的结果，并存储在元组 l1pm 中
        l1pm = (special.log1p(1), special.log1p(1.1), special.log1p(1.2))
        # 期望的 log 函数计算结果，存储在元组 l1pmrl 中
        l1pmrl = (log(2), log(2.1), log(2.2))
        # 断言 l1pm 与 l1pmrl 的元素几乎相等，精度为 8 位小数
        assert_array_almost_equal(l1pm, l1pmrl, 8)
class TestLegendreFunctions:
    # 定义测试类 TestLegendreFunctions，用于测试 Legendre 函数相关功能

    def test_clpmn(self):
        # 测试函数 test_clpmn，测试 clpmn 函数的返回值是否正确
        z = 0.5+0.3j
        # 设置复数变量 z
        clp = special.clpmn(2, 2, z, 3)
        # 调用 clpmn 函数计算 Legendre 函数并赋值给 clp
        assert_array_almost_equal(clp,
                   (array([[1.0000, z, 0.5*(3*z*z-1)],
                           [0.0000, sqrt(z*z-1), 3*z*sqrt(z*z-1)],
                           [0.0000, 0.0000, 3*(z*z-1)]]),
                    array([[0.0000, 1.0000, 3*z],
                           [0.0000, z/sqrt(z*z-1), 3*(2*z*z-1)/sqrt(z*z-1)],
                           [0.0000, 0.0000, 6*z]])),
                    7)
        # 使用 assert_array_almost_equal 断言 clp 的值与期望值数组的近似性，精确到小数点后 7 位

    def test_clpmn_close_to_real_2(self):
        # 测试函数 test_clpmn_close_to_real_2，测试 clpmn 函数在接近实数值时的返回值
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        # 设置变量 eps, m, n, x
        clp_plus = special.clpmn(m, n, x+1j*eps, 2)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 2)[0][m, n]
        # 调用 clpmn 函数计算 Legendre 函数，分别使用实部增加和减少 eps 值的复数
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x),
                                         special.lpmv(m, n, x)]),
                                  7)
        # 使用 assert_array_almost_equal 断言 clp_plus 和 clp_minus 的值与期望值数组的近似性，精确到小数点后 7 位

    def test_clpmn_close_to_real_3(self):
        # 测试函数 test_clpmn_close_to_real_3，测试 clpmn 函数在接近实数值时的返回值（带相位因子）
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        # 设置变量 eps, m, n, x
        clp_plus = special.clpmn(m, n, x+1j*eps, 3)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 3)[0][m, n]
        # 调用 clpmn 函数计算 Legendre 函数，分别使用实部增加和减少 eps 值的复数，并带有相位因子
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x)*np.exp(-0.5j*m*np.pi),
                                         special.lpmv(m, n, x)*np.exp(0.5j*m*np.pi)]),
                                  7)
        # 使用 assert_array_almost_equal 断言 clp_plus 和 clp_minus 的值与期望值数组的近似性，精确到小数点后 7 位

    def test_clpmn_across_unit_circle(self):
        # 测试函数 test_clpmn_across_unit_circle，测试 clpmn 函数在单位圆周围的行为
        eps = 1e-7
        m = 1
        n = 1
        x = 1j
        # 设置变量 eps, m, n, x
        for type in [2, 3]:
            # 循环遍历 type 取值为 2 和 3
            assert_almost_equal(special.clpmn(m, n, x+1j*eps, type)[0][m, n],
                            special.clpmn(m, n, x-1j*eps, type)[0][m, n], 6)
            # 使用 assert_almost_equal 断言 clpmn 函数在单位圆周围两个复数点的返回值近似性，精确到小数点后 6 位

    def test_inf(self):
        # 测试函数 test_inf，测试 clpmn 函数在特定 z 和 n,m 值时的返回值是否为无穷大
        for z in (1, -1):
            # 循环遍历 z 取值为 1 和 -1
            for n in range(4):
                # 循环遍历 n 取值范围为 0 到 3
                for m in range(1, n):
                    # 循环遍历 m 取值范围为 1 到 n-1
                    lp = special.clpmn(m, n, z)
                    # 调用 clpmn 函数计算 Legendre 函数
                    assert_(np.isinf(lp[1][1,1:]).all())
                    # 使用 assert_ 断言 lp 的返回值的某些部分是否全部为无穷大
                    lp = special.lpmn(m, n, z)
                    # 调用 lpmn 函数计算 Legendre 函数

    def test_deriv_clpmn(self):
        # 测试函数 test_deriv_clpmn，测试 clpmn 函数的导数计算是否准确
        # data inside and outside of the unit circle
        # 单位圆内外的数据
        zvals = [0.5+0.5j, -0.5+0.5j, -0.5-0.5j, 0.5-0.5j,
                 1+1j, -1+1j, -1-1j, 1-1j]
        # 设置复数列表 zvals
        m = 2
        n = 3
        # 设置整数变量 m 和 n
        for type in [2, 3]:
            # 循环遍历 type 取值为 2 和 3
            for z in zvals:
                # 循环遍历 zvals 中的复数 z
                for h in [1e-3, 1e-3j]:
                    # 循环遍历 h 列表中的实数和虚数
                    approx_derivative = (special.clpmn(m, n, z+0.5*h, type)[0]
                                         - special.clpmn(m, n, z-0.5*h, type)[0])/h
                    # 计算 clpmn 函数在 z 处的数值导数的近似值
                    assert_allclose(special.clpmn(m, n, z, type)[1],
                                    approx_derivative,
                                    rtol=1e-4)
                    # 使用 assert_allclose 断言 clpmn 函数在 z 处的导数与近似值的接近性，相对误差为 1e-4
    def test_lpmn(self):
        # 调用 special 模块的 lpmn 函数计算 Legendre 函数 P(0, 2) 的值，参数为 (0, 2, 0.5)
        lp = special.lpmn(0, 2, 0.5)
        # 使用 assert_array_almost_equal 断言 lp 的返回值与给定的数值数组接近，精度为 4 位小数
        assert_array_almost_equal(lp, (array([[1.00000,
                                              0.50000,
                                              -0.12500]]),
                                       array([[0.00000,
                                              1.00000,
                                              1.50000]])), 4)

    def test_lpn(self):
        # 调用 special 模块的 lpn 函数计算 Legendre 函数 P_n(2, 0.5) 的值，参数为 (2, 0.5)
        lpnf = special.lpn(2, 0.5)
        # 使用 assert_array_almost_equal 断言 lpnf 的返回值与给定的数值数组接近，精度为 4 位小数
        assert_array_almost_equal(lpnf, (array([1.00000,
                                                0.50000,
                                                -0.12500]),
                                         array([0.00000,
                                                1.00000,
                                                1.50000])), 4)

    def test_lpmv(self):
        # 调用 special 模块的 lpmv 函数计算 Associated Legendre 函数 P(0, 2, 0.5) 的值，参数为 (0, 2, 0.5)
        lp = special.lpmv(0, 2, 0.5)
        # 使用 assert_almost_equal 断言 lp 的返回值与给定的数值接近，精度为 7 位小数
        assert_almost_equal(lp, -0.125, 7)
        # 调用 special 模块的 lpmv 函数计算 Associated Legendre 函数 P(0, 40, 0.001) 的值，参数为 (0, 40, 0.001)
        lp = special.lpmv(0, 40, 0.001)
        # 使用 assert_almost_equal 断言 lp 的返回值与给定的数值接近，精度为 7 位小数
        assert_almost_equal(lp, 0.1252678976534484, 7)

        # XXX: 这个测试超出了当前实现的定义域，
        #      所以确保它返回 NaN 而不是错误答案。
        # 使用 np.errstate 防止所有错误，然后调用 special 模块的 lpmv 函数计算 Associated Legendre 函数 P(-1, -1, 0.001) 的值，参数为 (-1, -1, 0.001)
        with np.errstate(all='ignore'):
            lp = special.lpmv(-1, -1, 0.001)
        # 使用 assert_ 断言 lp 不等于 0 或者 lp 是 NaN
        assert_(lp != 0 or np.isnan(lp))

    def test_lqmn(self):
        # 调用 special 模块的 lqmn 函数计算 Legendre 函数 Q_mn(0, 2, 0.5) 的值，参数为 (0, 2, 0.5)
        lqmnf = special.lqmn(0, 2, 0.5)
        # 调用 special 模块的 lqn 函数计算 Legendre 函数 Q_n(2, 0.5) 的值，参数为 (2, 0.5)
        lqf = special.lqn(2, 0.5)
        # 使用 assert_array_almost_equal 断言 lqmnf 的第一个元素的第一个值与 lqf 的第一个元素的值接近，精度为 4 位小数
        assert_array_almost_equal(lqmnf[0][0], lqf[0], 4)
        # 使用 assert_array_almost_equal 断言 lqmnf 的第二个元素的第一个值与 lqf 的第二个元素的值接近，精度为 4 位小数
        assert_array_almost_equal(lqmnf[1][0], lqf[1], 4)

    def test_lqmn_gt1(self):
        """实数参数大于 1.0001 时算法会发生变化
           对 m=2, n=1 的分析结果进行测试
        """
        # 设置测试用例的参数
        x0 = 1.0001
        delta = 0.00002
        # 遍历测试用例中的两个参数值
        for x in (x0 - delta, x0 + delta):
            # 调用 special 模块的 lqmn 函数计算 Legendre 函数 Q_mn(2, 1, x) 的值，并获取最后一个元素的值
            lq = special.lqmn(2, 1, x)[0][-1, -1]
            # 计算预期的分析结果
            expected = 2 / (x * x - 1)
            # 使用 assert_almost_equal 断言 lq 的值与预期值接近
            assert_almost_equal(lq, expected)

    def test_lqmn_shape(self):
        # 调用 special 模块的 lqmn 函数计算 Legendre 函数 Q_mn(4, 4, 1.1) 的值，并获取返回的两个数组
        a, b = special.lqmn(4, 4, 1.1)
        # 使用 assert_equal 断言 a 的形状与预期的形状相同
        assert_equal(a.shape, (5, 5))
        # 使用 assert_equal 断言 b 的形状与预期的形状相同
        assert_equal(b.shape, (5, 5))

        # 调用 special 模块的 lqmn 函数计算 Legendre 函数 Q_mn(4, 0, 1.1) 的值，并获取返回的两个数组
        a, b = special.lqmn(4, 0, 1.1)
        # 使用 assert_equal 断言 a 的形状与预期的形状相同
        assert_equal(a.shape, (5, 1))
        # 使用 assert_equal 断言 b 的形状与预期的形状相同
        assert_equal(b.shape, (5, 1))

    def test_lqn(self):
        # 调用 special 模块的 lqn 函数计算 Legendre 函数 Q_n(2, 0.5) 的值，参数为 (2, 0.5)
        lqf = special.lqn(2, 0.5)
        # 使用 assert_array_almost_equal 断言 lqf 的返回值与给定的数值数组接近，精度为 4 位小数
        assert_array_almost_equal(lqf, (array([0.5493, -0.7253, -0.8187]),
                                       array([1.3333, 1.216, -0.8427])), 4)

    @pytest.mark.parametrize("function", [special.lpn, special.lqn])
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("z_complex", [False, True])
    @pytest.mark.parametrize("z_inexact", [False, True])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (), (1, ), (2, ), (2, 1), (1, 2), (2, 2), (2, 2, 1), (2, 2, 2)
        ]
    )
    # 定义测试函数，用于测试带有数组输入的函数的输出形状
    def test_array_inputs_lxn(self, function, n, z_complex, z_inexact, input_shape):
        """Tests for correct output shapes."""
        # 使用固定种子创建随机数生成器
        rng = np.random.default_rng(1234)
        # 根据 z_inexact 参数选择不同的随机数生成方式
        if z_inexact:
            z = rng.integers(-3, 3, size=input_shape)
        else:
            z = rng.uniform(-1, 1, size=input_shape)

        # 如果 z_complex 为 True，则将 z 转换为复数
        if z_complex:
            z = 1j * z + 0.5j * z

        # 调用待测试的函数 function，获取输出 P_z 和 P_d_z
        P_z, P_d_z = function(n, z)
        # 断言输出 P_z 和 P_d_z 的形状是否符合预期
        assert P_z.shape == (n + 1, ) + input_shape
        assert P_d_z.shape == (n + 1, ) + input_shape

    # 使用 pytest 的参数化标记定义多组测试参数，针对特定的函数 function 和输入参数组合进行测试
    @pytest.mark.parametrize("function", [special.lqmn])
    @pytest.mark.parametrize(
        "m,n",
        [(0, 1), (1, 2), (1, 4), (3, 8), (11, 16), (19, 32)]
    )
    @pytest.mark.parametrize("z_inexact", [False, True])
    @pytest.mark.parametrize(
        "input_shape", [
            (), (1, ), (2, ), (2, 1), (1, 2), (2, 2), (2, 2, 1)
        ]
    )
    # 定义测试函数，用于测试带有数组输入的函数的输出形状和数据类型
    def test_array_inputs_lxmn(self, function, m, n, z_inexact, input_shape):
        """Tests for correct output shapes and dtypes."""
        # 使用固定种子创建随机数生成器
        rng = np.random.default_rng(1234)
        # 根据 z_inexact 参数选择不同的随机数生成方式
        if z_inexact:
            z = rng.integers(-3, 3, size=input_shape)
        else:
            z = rng.uniform(-1, 1, size=input_shape)

        # 调用待测试的函数 function，获取输出 P_z 和 P_d_z
        P_z, P_d_z = function(m, n, z)
        # 断言输出 P_z 和 P_d_z 的形状是否符合预期
        assert P_z.shape == (m + 1, n + 1) + input_shape
        assert P_d_z.shape == (m + 1, n + 1) + input_shape

    # 使用 pytest 的参数化标记定义多组测试参数，针对特定的函数 function 和输入参数组合进行测试
    @pytest.mark.parametrize("function", [special.clpmn, special.lqmn])
    @pytest.mark.parametrize(
        "m,n",
        [(0, 1), (1, 2), (1, 4), (3, 8), (11, 16), (19, 32)]
    )
    @pytest.mark.parametrize(
        "input_shape", [
            (), (1, ), (2, ), (2, 1), (1, 2), (2, 2), (2, 2, 1)
        ]
    )
    # 定义测试函数，用于测试带有复数数组输入的函数的输出形状和数据类型
    def test_array_inputs_clxmn(self, function, m, n, input_shape):
        """Tests for correct output shapes and dtypes."""
        # 使用固定种子创建随机数生成器
        rng = np.random.default_rng(1234)
        # 使用均匀分布生成复数数组 z
        z = rng.uniform(-1, 1, size=input_shape)
        z = 1j * z + 0.5j * z

        # 调用待测试的函数 function，获取输出 P_z 和 P_d_z
        P_z, P_d_z = function(m, n, z)
        # 断言输出 P_z 和 P_d_z 的形状是否符合预期
        assert P_z.shape == (m + 1, n + 1) + input_shape
        assert P_d_z.shape == (m + 1, n + 1) + input_shape
class TestMathieu:
    # Mathieu 测试类

    def test_mathieu_a(self):
        # mathieu_a 方法，暂未实现
        pass

    def test_mathieu_even_coef(self):
        # 调用 special 模块的 mathieu_even_coef 函数，参数为 (2, 5)
        special.mathieu_even_coef(2, 5)
        # Q not defined broken and cannot figure out proper reporting order
        # Q 未定义，无法确定正确的报告顺序

    def test_mathieu_odd_coef(self):
        # mathieu_odd_coef 方法，暂未实现
        # 与上述问题相同
        pass


class TestFresnelIntegral:
    # Fresnel 积分测试类

    def test_modfresnelp(self):
        # modfresnelp 方法，暂未实现
        pass

    def test_modfresnelm(self):
        # modfresnelm 方法，暂未实现
        pass


class TestOblCvSeq:
    # OblCvSeq 测试类

    def test_obl_cv_seq(self):
        # 调用 special 模块的 obl_cv_seq 函数，参数为 (0, 3, 1)
        obl = special.obl_cv_seq(0, 3, 1)
        # 断言 obl 结果与给定数组几乎相等，精度为小数点后五位
        assert_array_almost_equal(obl, array([-0.348602,
                                              1.393206,
                                              5.486800,
                                              11.492120]), 5)


class TestParabolicCylinder:
    # Parabolic Cylinder 测试类

    def test_pbdn_seq(self):
        # 调用 special 模块的 pbdn_seq 函数，参数为 (1, 0.1)
        pb = special.pbdn_seq(1, 0.1)
        # 断言 pb 结果与给定数组几乎相等，精度为小数点后四位
        assert_array_almost_equal(pb, (array([0.9975,
                                              0.0998]),
                                      array([-0.0499,
                                             0.9925])), 4)

    def test_pbdv(self):
        # 调用 special 模块的 pbdv 函数，参数为 (1, 0.2)
        special.pbdv(1, 0.2)
        # 计算 pbdv 函数的一部分并返回结果

    def test_pbdv_seq(self):
        # 调用 special 模块的 pbdn_seq 函数，参数为 (1, 0.1)
        pbn = special.pbdn_seq(1, 0.1)
        # 调用 special 模块的 pbdv_seq 函数，参数为 (1, 0.1)
        pbv = special.pbdv_seq(1, 0.1)
        # 断言 pbv 结果与 pbn 的实部几乎相等，精度为小数点后四位
        assert_array_almost_equal(pbv, (real(pbn[0]), real(pbn[1])), 4)

    def test_pbdv_points(self):
        # 简单情况
        eta = np.linspace(-10, 10, 5)
        z = 2**(eta/2) * np.sqrt(np.pi) / special.gamma(0.5 - 0.5 * eta)
        # 断言 pbdv 函数返回结果与 z 几乎相等，相对容差为 1e-14，绝对容差为 1e-14
        assert_allclose(special.pbdv(eta, 0.)[0], z, rtol=1e-14, atol=1e-14)

        # 一些点
        # 断言 pbdv 函数返回结果与给定值几乎相等，相对容差为 1e-12
        assert_allclose(special.pbdv(10.34, 20.44)[0], 1.3731383034455e-32, rtol=1e-12)
        assert_allclose(special.pbdv(-9.53, 3.44)[0], 3.166735001119246e-8, rtol=1e-12)

    def test_pbdv_gradient(self):
        x = np.linspace(-4, 4, 8)[:, None]
        eta = np.linspace(-10, 10, 5)[None, :]

        p = special.pbdv(eta, x)
        eps = 1e-7 + 1e-7 * abs(x)
        dp = (special.pbdv(eta, x + eps)[0] - special.pbdv(eta, x - eps)[0]) / eps / 2.
        # 断言 p 的第二部分与 dp 几乎相等，相对容差为 1e-6，绝对容差为 1e-6
        assert_allclose(p[1], dp, rtol=1e-6, atol=1e-6)

    def test_pbvv_gradient(self):
        x = np.linspace(-4, 4, 8)[:, None]
        eta = np.linspace(-10, 10, 5)[None, :]

        p = special.pbvv(eta, x)
        eps = 1e-7 + 1e-7 * abs(x)
        dp = (special.pbvv(eta, x + eps)[0] - special.pbvv(eta, x - eps)[0]) / eps / 2.
        # 断言 p 的第二部分与 dp 几乎相等，相对容差为 1e-6，绝对容差为 1e-6
        assert_allclose(p[1], dp, rtol=1e-6, atol=1e-6)

    def test_pbvv_seq(self):
        # 调用 special 模块的 pbvv_seq 函数，参数为 (2, 3)
        res1, res2 = special.pbvv_seq(2, 3)
        # 断言 res1 与给定数组几乎相等
        assert_allclose(res1, np.array([2.976319645712036,
                                        1.358840996329579,
                                        0.5501016716383508]))
        # 断言 res2 与给定数组几乎相等
        assert_allclose(res2, np.array([3.105638472238475,
                                        0.9380581512176672,
                                        0.533688488872053]))


class TestPolygamma:
    # Polygamma 测试类
    # 来自 A&S 第 6.2 表格 (第 271 页)
    # 定义一个测试方法，用于测试 polygamma 函数的行为
    def test_polygamma(self):
        # 计算 polygamma(2, 1) 的值
        poly2 = special.polygamma(2, 1)
        # 计算 polygamma(3, 1) 的值
        poly3 = special.polygamma(3, 1)
        # 断言 poly2 的计算结果接近 -2.4041138063，精度为小数点后10位
        assert_almost_equal(poly2, -2.4041138063, 10)
        # 断言 poly3 的计算结果接近 6.4939394023，精度为小数点后10位
        assert_almost_equal(poly3, 6.4939394023, 10)

        # 测试 polygamma(0, x) 是否等于 psi(x)
        x = [2, 3, 1.1e14]
        # 断言 polygamma(0, x) 的计算结果接近 psi(x)
        assert_almost_equal(special.polygamma(0, x), special.psi(x))

        # 测试广播功能
        n = [0, 1, 2]
        x = [0.5, 1.5, 2.5]
        expected = [-1.9635100260214238, 0.93480220054467933,
                    -0.23620405164172739]
        # 断言 polygamma(n, x) 的计算结果接近 expected 数组
        assert_almost_equal(special.polygamma(n, x), expected)
        # 使用 np.vstack 创建 expected 的两倍重复数组，测试广播
        expected = np.vstack([expected]*2)
        # 断言 polygamma(n, np.vstack([x]*2)) 的计算结果接近 expected
        assert_almost_equal(special.polygamma(n, np.vstack([x]*2)),
                            expected)
        # 断言 polygamma(np.vstack([n]*2), x) 的计算结果接近 expected
        assert_almost_equal(special.polygamma(np.vstack([n]*2), x),
                            expected)
class TestProCvSeq:
    # 定义测试类 TestProCvSeq
    def test_pro_cv_seq(self):
        # 调用 special 模块中的 pro_cv_seq 函数，生成 prol 序列
        prol = special.pro_cv_seq(0,3,1)
        # 使用 assert_array_almost_equal 函数检查 prol 序列是否近似等于给定数组
        assert_array_almost_equal(prol,array([0.319000,
                                               2.593084,
                                               6.533471,
                                               12.514462]),5)


class TestPsi:
    # 定义测试类 TestPsi
    def test_psi(self):
        # 调用 special 模块中的 psi 函数，计算 ps 值
        ps = special.psi(1)
        # 使用 assert_almost_equal 函数检查 ps 值是否近似等于给定值
        assert_almost_equal(ps,-0.57721566490153287,8)


class TestRadian:
    # 定义测试类 TestRadian
    def test_radian(self):
        # 调用 special 模块中的 radian 函数，计算 rad 值
        rad = special.radian(90,0,0)
        # 使用 assert_almost_equal 函数检查 rad 值是否近似等于 pi/2.0
        assert_almost_equal(rad,pi/2.0,5)

    def test_radianmore(self):
        # 调用 special 模块中的 radian 函数，计算 rad1 值
        rad1 = special.radian(90,1,60)
        # 使用 assert_almost_equal 函数检查 rad1 值是否近似等于 pi/2 + 0.0005816135199345904
        assert_almost_equal(rad1,pi/2+0.0005816135199345904,5)


class TestRiccati:
    # 定义测试类 TestRiccati
    def test_riccati_jn(self):
        # 设定变量 N 和 x
        N, x = 2, 0.2
        # 创建一个空的 N x N 的数组 S
        S = np.empty((N, N))
        # 循环计算球面贝塞尔函数及其导数，并填充数组 S
        for n in range(N):
            j = special.spherical_jn(n, x)
            jp = special.spherical_jn(n, x, derivative=True)
            S[0,n] = x*j
            S[1,n] = x*jp + j
        # 使用 assert_array_almost_equal 函数检查数组 S 是否近似等于 riccati_jn 函数返回的结果
        assert_array_almost_equal(S, special.riccati_jn(n, x), 8)

    def test_riccati_yn(self):
        # 设定变量 N 和 x
        N, x = 2, 0.2
        # 创建一个空的 N x N 的数组 C
        C = np.empty((N, N))
        # 循环计算球面贝塞尔函数及其导数，并填充数组 C
        for n in range(N):
            y = special.spherical_yn(n, x)
            yp = special.spherical_yn(n, x, derivative=True)
            C[0,n] = x*y
            C[1,n] = x*yp + y
        # 使用 assert_array_almost_equal 函数检查数组 C 是否近似等于 riccati_yn 函数返回的结果
        assert_array_almost_equal(C, special.riccati_yn(n, x), 8)


class TestRound:
    # 定义测试类 TestRound
    def test_round(self):
        # 调用 special 模块中的 round 函数，分别对 10.1, 10.4, 10.5, 10.6 进行四舍五入
        rnd = list(map(int, (special.round(10.1),
                             special.round(10.4),
                             special.round(10.5),
                             special.round(10.6))))

        # 按照文档说明，special.round 函数对于 10.5 应向最接近的偶数舍入
        # 在某些平台上，这可能无法正常工作，因此测试可能会失败
        # 不过，此单元测试是正确编写的
        # 预期的四舍五入结果
        rndrl = (10,10,10,11)
        # 使用 assert_array_equal 函数检查四舍五入结果是否与预期相符
        assert_array_equal(rnd,rndrl)


def test_sph_harm():
    # 从 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics 中提取的测试数据
    # 测试 special 模块中的 sph_harm 函数
    sh = special.sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    # 使用 assert_array_almost_equal 函数检查 sph_harm 函数的返回值是否近似等于给定值
    assert_array_almost_equal(sh(0,0,0,0),
           0.5/sqrt(pi))
    assert_array_almost_equal(sh(-2,2,0.,pi/4),
           0.25*sqrt(15./(2.*pi)) *
           (sin(pi/4))**2.)
    assert_array_almost_equal(sh(-2,2,0.,pi/2),
           0.25*sqrt(15./(2.*pi)))
    assert_array_almost_equal(sh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi)) *
           exp(0+2.*pi*1j)*sin(pi/2.)**2.)
    assert_array_almost_equal(sh(2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi)) *
           exp(0+2.*pi/4.*1j) *
           sin(pi/3.)**2. *
           (7.*cos(pi/3.)**2.-1))
    assert_array_almost_equal(sh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi)) *
           exp(0+4.*pi/8.*1j)*sin(pi/6.)**4.)
def test_sph_harm_ufunc_loop_selection():
    # 解决 GitHub 上的问题 https://github.com/scipy/scipy/issues/4895
    dt = np.dtype(np.complex128)
    # 断言特定参数下特定函数的返回值类型为复数复128位
    assert_equal(special.sph_harm(0, 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm([0], 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, [0], 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, [0], 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, 0, [0]).dtype, dt)
    assert_equal(special.sph_harm([0], [0], [0], [0]).dtype, dt)


class TestStruve:
    def _series(self, v, z, n=100):
        """计算 Struve 函数及其幂级数的误差估计。"""
        k = arange(0, n)
        r = (-1)**k * (.5*z)**(2*k+v+1)/special.gamma(k+1.5)/special.gamma(k+v+1.5)
        err = abs(r).max() * finfo(double).eps * n
        return r.sum(), err

    def test_vs_series(self):
        """检查 Struve 函数与其幂级数的对比"""
        for v in [-20, -10, -7.99, -3.4, -1, 0, 1, 3.4, 12.49, 16]:
            for z in [1, 10, 19, 21, 30]:
                value, err = self._series(v, z)
                assert_allclose(special.struve(v, z), value, rtol=0, atol=err), (v, z)

    def test_some_values(self):
        assert_allclose(special.struve(-7.99, 21), 0.0467547614113, rtol=1e-7)
        assert_allclose(special.struve(-8.01, 21), 0.0398716951023, rtol=1e-8)
        assert_allclose(special.struve(-3.0, 200), 0.0142134427432, rtol=1e-12)
        assert_allclose(special.struve(-8.0, -41), 0.0192469727846, rtol=1e-11)
        assert_equal(special.struve(-12, -41), -special.struve(-12, 41))
        assert_equal(special.struve(+12, -41), -special.struve(+12, 41))
        assert_equal(special.struve(-11, -41), +special.struve(-11, 41))
        assert_equal(special.struve(+11, -41), +special.struve(+11, 41))

        assert_(isnan(special.struve(-7.1, -1)))
        assert_(isnan(special.struve(-10.1, -1)))

    def test_regression_679(self):
        """针对问题 #679 的回归测试"""
        assert_allclose(special.struve(-1.0, 20 - 1e-8),
                        special.struve(-1.0, 20 + 1e-8))
        assert_allclose(special.struve(-2.0, 20 - 1e-8),
                        special.struve(-2.0, 20 + 1e-8))
        assert_allclose(special.struve(-4.3, 20 - 1e-8),
                        special.struve(-4.3, 20 + 1e-8))


def test_chi2_smalldf():
    assert_almost_equal(special.chdtr(0.6,3), 0.957890536704110)


def test_ch2_inf():
    assert_equal(special.chdtr(0.7,np.inf), 1.0)


def test_chi2c_smalldf():
    assert_almost_equal(special.chdtrc(0.6,3), 1-0.957890536704110)


def test_chi2_inv_smalldf():
    assert_almost_equal(special.chdtri(0.6,1-0.957890536704110), 3)


def test_agm_simple():
    rtol = 1e-13

    # 高斯常数
    assert_allclose(1/special.agm(1, np.sqrt(2)), 0.834626841674073186,
                    rtol=rtol)

    # 这些值是使用 Wolfram Alpha 计算得出的，
    # 使用函数 ArithmeticGeometricMean[a, b]
    agm13 = 1.863616783244897
    # 定义常数 agm15 和 agm35，分别赋值为特定浮点数
    agm15 = 2.604008190530940
    agm35 = 3.936235503649555
    
    # 使用 assert_allclose 函数检查 special.agm 的计算结果是否与预期接近
    assert_allclose(special.agm([[1], [3]], [1, 3, 5]),
                    [[1, agm13, agm15],
                     [agm13, 3, agm35]], rtol=rtol)
    
    # 计算通过 mpmath 的迭代公式得到的 agm12 值，精度为 1000 位
    agm12 = 1.4567910310469068
    
    # 使用 assert_allclose 函数检查特定参数下 special.agm 的计算结果是否与 agm12 接近
    assert_allclose(special.agm(1, 2), agm12, rtol=rtol)
    assert_allclose(special.agm(2, 1), agm12, rtol=rtol)
    assert_allclose(special.agm(-1, -2), -agm12, rtol=rtol)
    assert_allclose(special.agm(24, 6), 13.458171481725614, rtol=rtol)
    assert_allclose(special.agm(13, 123456789.5), 11111458.498599306,
                    rtol=rtol)
    assert_allclose(special.agm(1e30, 1), 2.229223055945383e+28, rtol=rtol)
    assert_allclose(special.agm(1e-22, 1), 0.030182566420169886, rtol=rtol)
    assert_allclose(special.agm(1e150, 1e180), 2.229223055945383e+178,
                    rtol=rtol)
    assert_allclose(special.agm(1e180, 1e-150), 2.0634722510162677e+177,
                    rtol=rtol)
    assert_allclose(special.agm(1e-150, 1e-170), 3.3112619670463756e-152,
                    rtol=rtol)
    
    # 获取浮点数的最小和最大规格化数
    fi = np.finfo(1.0)
    
    # 使用 assert_allclose 函数检查特定参数下 special.agm 的计算结果是否与预期接近
    assert_allclose(special.agm(fi.tiny, fi.max), 1.9892072050015473e+305,
                    rtol=rtol)
    assert_allclose(special.agm(0.75*fi.max, fi.max), 1.564904312298045e+308,
                    rtol=rtol)
    assert_allclose(special.agm(fi.tiny, 3*fi.tiny), 4.1466849866735005e-308,
                    rtol=rtol)
    
    # 零、NaN 和无穷大的特殊情况的断言检查
    assert_equal(special.agm(0, 0), 0)
    assert_equal(special.agm(99, 0), 0)
    assert_equal(special.agm(-1, 10), np.nan)
    assert_equal(special.agm(0, np.inf), np.nan)
    assert_equal(special.agm(np.inf, 0), np.nan)
    assert_equal(special.agm(0, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, 0), np.nan)
    assert_equal(special.agm(np.inf, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, np.inf), np.nan)
    assert_equal(special.agm(1, np.nan), np.nan)
    assert_equal(special.agm(np.nan, -1), np.nan)
    assert_equal(special.agm(1, np.inf), np.inf)
    assert_equal(special.agm(np.inf, 1), np.inf)
    assert_equal(special.agm(-1, -np.inf), -np.inf)
    assert_equal(special.agm(-np.inf, -1), -np.inf)
def test_legacy():
    # Legacy behavior: truncating arguments to integers
    # 使用 suppress_warnings 上下文管理器来捕获特定的警告
    with suppress_warnings() as sup:
        # 过滤特定的 RuntimeWarning，该警告是浮点数被截断为整数的警告
        sup.filter(RuntimeWarning, "floating point number truncated to an integer")
        # 断言特殊函数的结果相等，其中参数被截断为整数
        assert_equal(special.expn(1, 0.3), special.expn(1.8, 0.3))
        assert_equal(special.nbdtrc(1, 2, 0.3), special.nbdtrc(1.8, 2.8, 0.3))
        assert_equal(special.nbdtr(1, 2, 0.3), special.nbdtr(1.8, 2.8, 0.3))
        assert_equal(special.nbdtri(1, 2, 0.3), special.nbdtri(1.8, 2.8, 0.3))
        assert_equal(special.pdtri(1, 0.3), special.pdtri(1.8, 0.3))
        assert_equal(special.kn(1, 0.3), special.kn(1.8, 0.3))
        assert_equal(special.yn(1, 0.3), special.yn(1.8, 0.3))
        assert_equal(special.smirnov(1, 0.3), special.smirnov(1.8, 0.3))
        assert_equal(special.smirnovi(1, 0.3), special.smirnovi(1.8, 0.3))


@with_special_errors
def test_error_raising():
    # 断言调用特殊函数时会引发特定的异常
    assert_raises(special.SpecialFunctionError, special.iv, 1, 1e99j)


def test_xlogy():
    def xfunc(x, y):
        # 使用 np.errstate 来处理特定的浮点错误，这里忽略无效操作的警告
        with np.errstate(invalid='ignore'):
            # 如果 x 等于 0 并且 y 不是 NaN，则返回 x；否则返回 x * log(y)
            if x == 0 and not np.isnan(y):
                return x
            else:
                return x * np.log(y)

    # 创建包含特定数值对的 numpy 数组 z1 和 z2
    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0)], dtype=float)
    z2 = np.r_[z1, [(0, 1j), (1, 1j)]]

    # 使用向量化函数 np.vectorize 对 z1 和 z2 应用 xfunc 函数
    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    # 断言特殊函数的结果与预期值相等，使用特定的相对和绝对容差
    assert_func_equal(special.xlogy, w1, z1, rtol=1e-13, atol=1e-13)
    w2 = np.vectorize(xfunc)(z2[:,0], z2[:,1])
    assert_func_equal(special.xlogy, w2, z2, rtol=1e-13, atol=1e-13)


def test_xlog1py():
    def xfunc(x, y):
        with np.errstate(invalid='ignore'):
            if x == 0 and not np.isnan(y):
                return x
            else:
                return x * np.log1p(y)

    # 创建包含特定数值对的 numpy 数组 z1
    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0),
                     (1, 1e-30)], dtype=float)
    # 使用向量化函数 np.vectorize 对 z1 应用 xfunc 函数
    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    # 断言特殊函数的结果与预期值相等，使用特定的相对和绝对容差
    assert_func_equal(special.xlog1py, w1, z1, rtol=1e-13, atol=1e-13)


def test_entr():
    def xfunc(x):
        # 如果 x 小于 0，则返回负无穷；否则返回 -x*log(x)
        if x < 0:
            return -np.inf
        else:
            return -special.xlogy(x, x)
    
    # 创建包含特定值的 numpy 数组 z
    values = (0, 0.5, 1.0, np.inf)
    signs = [-1, 1]
    arr = []
    # 生成所有可能的符号和值的组合，添加到数组 arr 中
    for sgn, v in itertools.product(signs, values):
        arr.append(sgn * v)
    z = np.array(arr, dtype=float)
    # 使用向量化函数 np.vectorize 对 z 应用 xfunc 函数
    w = np.vectorize(xfunc, otypes=[np.float64])(z)
    # 断言特殊函数的结果与预期值相等，使用特定的相对和绝对容差
    assert_func_equal(special.entr, w, z, rtol=1e-13, atol=1e-13)


def test_kl_div():
    def xfunc(x, y):
        # 根据 x 和 y 的值返回特定的结果，以确保函数在自然域内保持凸性
        if x < 0 or y < 0 or (y == 0 and x != 0):
            return np.inf
        elif np.isposinf(x) or np.isposinf(y):
            return np.inf
        elif x == 0:
            return y
        else:
            return special.xlogy(x, x/y) - x + y
    
    # 创建包含特定值对的数组 arr
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    # 生成所有可能的符号和值的组合，添加到数组 arr 中
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    # 将列表 arr 转换为 NumPy 数组 z，并指定数据类型为 float
    z = np.array(arr, dtype=float)
    # 使用 np.vectorize 对函数 xfunc 进行向量化，指定输出类型为 np.float64，并对 z 的每一列进行操作
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    # 断言特殊函数 special.kl_div 和向量化后的结果 w 相等，指定比较的相对容差和绝对容差
    assert_func_equal(special.kl_div, w, z, rtol=1e-13, atol=1e-13)
def test_rel_entr():
    # 定义函数 xfunc，计算相对熵（KL 散度）中的元素值
    def xfunc(x, y):
        if x > 0 and y > 0:
            return special.xlogy(x, x/y)  # 若 x 和 y 都大于 0，则计算 x * log(x / y)
        elif x == 0 and y >= 0:
            return 0  # 若 x 为 0 且 y 大于等于 0，则返回 0
        else:
            return np.inf  # 其他情况返回无穷大
    # 定义 values 和 signs 变量
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    # 使用 itertools.product 对 signs 和 values 进行笛卡尔积，生成组合并添加到 arr 中
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    z = np.array(arr, dtype=float)
    # 对 xfunc 进行向量化处理，应用于 z 的列，并将结果保存到 w 中
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    # 使用 assert_func_equal 检查 special.rel_entr 函数的结果与 w 的一致性，设置相对和绝对容差
    assert_func_equal(special.rel_entr, w, z, rtol=1e-13, atol=1e-13)


def test_rel_entr_gh_20710_near_zero():
    # 检查非常接近的输入精度
    inputs = np.array([
        # x, y
        (0.9456657713430001, 0.9456657713430094),
        (0.48066098564791515, 0.48066098564794774),
        (0.786048657854401, 0.7860486578542367),
    ])
    # 使用 `x * mpmath.log(x / y)` 计算，精度为 30
    expected = [
        -9.325873406851269e-15,
        -3.258504577274724e-14,
        1.6431300764454033e-13,
    ]
    x = inputs[:, 0]
    y = inputs[:, 1]
    # 使用 assert_allclose 检查 special.rel_entr 的结果与预期结果的一致性，设置相对和绝对容差
    assert_allclose(special.rel_entr(x, y), expected, rtol=1e-13, atol=0)


def test_rel_entr_gh_20710_overflow():
    inputs = np.array([
        # x, y
        # 溢出情况
        (4, 2.22e-308),
        # 下溢情况
        (1e-200, 1e+200),
        # 亚正常情况
        (2.22e-308, 1e15),
    ])
    # 使用 `x * mpmath.log(x / y)` 计算，精度为 30
    expected = [
        2839.139983229607,
        -9.210340371976183e-198,
        -1.6493212008074475e-305,
    ]
    x = inputs[:, 0]
    y = inputs[:, 1]
    # 使用 assert_allclose 检查 special.rel_entr 的结果与预期结果的一致性，设置相对和绝对容差
    assert_allclose(special.rel_entr(x, y), expected, rtol=1e-13, atol=0)


def test_huber():
    # 使用 assert_equal 检查特定输入下 special.huber 的结果是否等于无穷大
    assert_equal(special.huber(-1, 1.5), np.inf)
    # 使用 assert_allclose 检查特定输入下 special.huber 的结果与预期结果的一致性，设置相对和绝对容差
    assert_allclose(special.huber(2, 1.5), 0.5 * np.square(1.5))
    assert_allclose(special.huber(2, 2.5), 2 * (2.5 - 0.5 * 2))

    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif np.abs(r) < delta:
            return 0.5 * np.square(r)
        else:
            return delta * (np.abs(r) - 0.5 * delta)

    z = np.random.randn(10, 2)
    # 对 xfunc 进行向量化处理，应用于 z 的列，并将结果保存到 w 中
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    # 使用 assert_func_equal 检查 special.huber 函数的结果与 w 的一致性，设置相对和绝对容差
    assert_func_equal(special.huber, w, z, rtol=1e-13, atol=1e-13)


def test_pseudo_huber():
    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif (not delta) or (not r):
            return 0
        else:
            return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)

    z = np.array(np.random.randn(10, 2).tolist() + [[0, 0.5], [0.5, 0]])
    # 对 xfunc 进行向量化处理，应用于 z 的列，并将结果保存到 w 中
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    # 使用 assert_func_equal 检查 special.pseudo_huber 函数的结果与 w 的一致性，设置相对和绝对容差
    assert_func_equal(special.pseudo_huber, w, z, rtol=1e-13, atol=1e-13)


def test_pseudo_huber_small_r():
    delta = 1.0
    r = 1e-18
    y = special.pseudo_huber(delta, r)
    # 使用 mpmath 计算预期值，并与实际结果进行比较
    #     import mpmath
    #     mpmath.mp.dps = 200
    #     r = mpmath.mpf(1e-18)
    #     expected = float(mpmath.sqrt(1 + r**2) - 1)
    # 预期的值，用科学计数法表示，表示接近零但不为零的浮点数
    expected = 5.0000000000000005e-37
    # 使用断言检查实际结果 y 是否与预期值 expected 接近，相对容差为 1e-13
    assert_allclose(y, expected, rtol=1e-13)
# 定义一个测试函数，用于测试运行时警告
def test_runtime_warning():
    # 断言在运行mathieu_odd_coef(1000, 1000)时会发出RuntimeWarning，并且警告消息包含'Too many predicted coefficients'
    with pytest.warns(RuntimeWarning, match=r'Too many predicted coefficients'):
        mathieu_odd_coef(1000, 1000)
    # 断言在运行mathieu_even_coef(1000, 1000)时会发出RuntimeWarning，并且警告消息包含'Too many predicted coefficients'
    with pytest.warns(RuntimeWarning, match=r'Too many predicted coefficients'):
        mathieu_even_coef(1000, 1000)


class TestStirling2:
    # 预定义的斯特林数第二类表格
    table = [
        [1],
        [0, 1],
        [0, 1, 1],
        [0, 1, 3, 1],
        [0, 1, 7, 6, 1],
        [0, 1, 15, 25, 10, 1],
        [0, 1, 31, 90, 65, 15, 1],
        [0, 1, 63, 301, 350, 140, 21, 1],
        [0, 1, 127, 966, 1701, 1050, 266, 28, 1],
        [0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1],
        [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1],
    ]

    # 参数化测试，用于测试斯特林数第二类的计算准确性
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_table_cases(self, is_exact, comp, kwargs):
        # 对每个 n 进行测试
        for n in range(1, len(self.table)):
            # k_values 是从 0 到 n 的列表
            k_values = list(range(n+1))
            # 取出第 n 行的预期结果
            row = self.table[n]
            # 比较计算结果与预期结果
            comp(row, stirling2([n], k_values, exact=is_exact), **kwargs)

    # 参数化测试，用于测试斯特林数第二类对单个整数的计算准确性
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_valid_single_integer(self, is_exact, comp, kwargs):
        # 测试斯特林数第二类计算结果与预期结果是否相等
        comp(stirling2(0, 0, exact=is_exact), self.table[0][0], **kwargs)
        comp(stirling2(4, 2, exact=is_exact), self.table[4][2], **kwargs)
        # 斯特林数第二类对于 (5, 3) 的计算结果应为 25
        comp(stirling2(5, 3, exact=is_exact), 25, **kwargs)
        # 斯特林数第二类对于 ([5], [3]) 的计算结果应为 [25]
        comp(stirling2([5], [3], exact=is_exact), [25], **kwargs)

    # 参数化测试，用于测试斯特林数第二类对负整数的处理
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_negative_integer(self, is_exact, comp, kwargs):
        # 对于负整数输入，斯特林数第二类的计算结果应为 0
        comp(stirling2(-1, -1, exact=is_exact), 0, **kwargs)
        comp(stirling2(-1, 2, exact=is_exact), 0, **kwargs)
        comp(stirling2(2, -1, exact=is_exact), 0, **kwargs)

    # 参数化测试，用于测试斯特林数第二类对数组输入的处理
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_array_inputs(self, is_exact, comp, kwargs):
        # 预期的结果数组
        ans = [self.table[10][3], self.table[10][4]]
        # 斯特林数第二类对数组输入的计算结果应与预期的结果数组相等
        comp(stirling2(asarray([10, 10]),
                       asarray([3, 4]),
                       exact=is_exact),
             ans)
        comp(stirling2([10, 10],
                       asarray([3, 4]),
                       exact=is_exact),
             ans)
        comp(stirling2(asarray([10, 10]),
                       [3, 4],
                       exact=is_exact),
             ans)
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-13})
    ])
    def test_mixed_values(self, is_exact, comp, kwargs):
        # 使用 pytest 的 parametrize 标记，定义多组参数化测试数据
        # 测试混合数值情况下的函数行为，预期结果存储在 ans 列表中
        ans = [0, 1, 3, 25, 1050, 5880, 9330]
        # 不同的 n 和 k 值用于测试函数 stirling2
        n = [-1, 0, 3, 5, 8, 10, 10]
        k = [-2, 0, 2, 3, 5, 7, 3]
        # 调用比较函数 comp，检查 stirling2 的返回结果是否符合预期
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
    
    def test_correct_parity(self):
        """Test parity follows well known identity.
    
        en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind#Parity
        """
        # 测试函数 stirling2 的奇偶性是否符合已知的数学恒等式
        n, K = 100, np.arange(101)
        # 断言 stirling2 的返回结果对 2 取模，应该等于根据 k 计算的结果对 2 取模
        assert_equal(
            stirling2(n, K, exact=True) % 2,
            [math.comb(n - (k // 2) - 1, n - k) % 2 for k in K],
        )
    
    def test_big_numbers(self):
        # 使用 mpmath 进行计算（大于 32 位）
        ans = asarray([48063331393110, 48004081105038305])
        n = [25, 30]
        k = [17, 4]
        # 断言 stirling2 的返回结果与预期结果 ans 相等
        assert array_equal(stirling2(n, k, exact=True), ans)
        # 大于 64 位的情况
        ans = asarray([2801934359500572414253157841233849412,
                       14245032222277144547280648984426251])
        n = [42, 43]
        k = [17, 23]
        # 断言 stirling2 的返回结果与预期结果 ans 相等
        assert array_equal(stirling2(n, k, exact=True), ans)
    
    @pytest.mark.parametrize("N", [4.5, 3., 4+1j, "12", np.nan])
    @pytest.mark.parametrize("K", [3.5, 3, "2", None])
    @pytest.mark.parametrize("is_exact", [True, False])
    def test_unsupported_input_types(self, N, K, is_exact):
        # 对象、浮点数、字符串、复数类型不被支持，会引发 TypeError
        with pytest.raises(TypeError):
            stirling2(N, K, exact=is_exact)
    
    @pytest.mark.parametrize("is_exact", [True, False])
    def test_numpy_array_int_object_dtype(self, is_exact):
        # Python 整数的任意精度在 numpy 数组中作为对象类型不被允许
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=object)
        k = asarray([1, 2, 3, 4], dtype=object)
        # 使用 pytest 断言会引发 TypeError
        with pytest.raises(TypeError):
            array_equal(stirling2(n, k, exact=is_exact), ans)
    
    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-13})
    ])
    def test_numpy_array_unsigned_int_dtype(self, is_exact, comp, kwargs):
        # numpy 无符号整数作为 numpy 数组的 dtype 是被允许的
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=np_ulong)
        k = asarray([1, 2, 3, 4], dtype=np_ulong)
        # 使用比较函数 comp 检查 stirling2 的返回结果是否符合预期
        comp(stirling2(n, k, exact=False), ans, **kwargs)
    def test_broadcasting_arrays_correctly(self, is_exact, comp, kwargs):
        # broadcasting is handled by stirling2
        # test leading 1s are replicated
        ans = asarray([[1, 15, 25, 10], [1, 7, 6, 1]])  # shape (2,4)
        n = asarray([[5, 5, 5, 5], [4, 4, 4, 4]])  # shape (2,4)
        k = asarray([1, 2, 3, 4])  # shape (4,)
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
        # test that dims both mismatch broadcast correctly (5,1) & (6,)
        n = asarray([[4], [4], [4], [4], [4]])
        k = asarray([0, 1, 2, 3, 4, 5])
        ans = asarray([[0, 1, 7, 6, 1, 0] for _ in range(5)])
        comp(stirling2(n, k, exact=False), ans, **kwargs)

    def test_temme_rel_max_error(self):
        # python integers with arbitrary precision are *not* allowed as
        # object type in numpy arrays are inconsistent from api perspective
        x = list(range(51, 101, 5))
        for n in x:
            k_entries = list(range(1, n+1))
            denom = stirling2([n], k_entries, exact=True)
            num = denom - stirling2([n], k_entries, exact=False)
            assert np.max(np.abs(num / denom)) < 2e-5



# 定义测试函数，用于验证数组广播的正确性
def test_broadcasting_arrays_correctly(self, is_exact, comp, kwargs):
    # stirling2 函数处理数组广播
    # 测试前导的1被复制
    ans = asarray([[1, 15, 25, 10], [1, 7, 6, 1]])  # 形状为 (2,4)
    n = asarray([[5, 5, 5, 5], [4, 4, 4, 4]])  # 形状为 (2,4)
    k = asarray([1, 2, 3, 4])  # 形状为 (4,)
    comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
    # 测试两个维度不匹配时的广播 (5,1) & (6,)
    n = asarray([[4], [4], [4], [4], [4]])
    k = asarray([0, 1, 2, 3, 4, 5])
    ans = asarray([[0, 1, 7, 6, 1, 0] for _ in range(5)])
    comp(stirling2(n, k, exact=False), ans, **kwargs)

# 定义测试函数，验证 Temme 算法的相对最大误差
def test_temme_rel_max_error(self):
    # Python 的整数具有任意精度，不允许作为 numpy 数组的对象类型
    x = list(range(51, 101, 5))
    for n in x:
        k_entries = list(range(1, n+1))
        denom = stirling2([n], k_entries, exact=True)
        num = denom - stirling2([n], k_entries, exact=False)
        assert np.max(np.abs(num / denom)) < 2e-5
```