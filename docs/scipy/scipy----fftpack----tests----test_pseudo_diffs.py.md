# `D:\src\scipysrc\scipy\scipy\fftpack\tests\test_pseudo_diffs.py`

```
# Created by Pearu Peterson, September 2002
# 由Pearu Peterson于2002年9月创建

__usage__ = """
Build fftpack:
  python setup_fftpack.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.fftpack.test(<level>)'
Run tests if fftpack is not installed:
  python tests/test_pseudo_diffs.py [<level>]
"""
# __usage__ 是一个多行字符串，提供了关于如何构建fftpack和运行测试的说明

from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)
# 导入 numpy.testing 模块中的断言函数 assert_equal, assert_almost_equal,
# assert_array_almost_equal，用于测试数值的相等性和近似性

from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
                           ihilbert, shift, fftfreq, cs_diff, sc_diff,
                           ss_diff, cc_diff)
# 导入 scipy.fftpack 模块中的一些函数和类，用于处理傅里叶变换和频谱分析

import numpy as np
# 导入 numpy 库，并使用 np 作为别名

from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
# 导入 numpy 库中的一些数学函数和常数，如等差数组生成、三角函数、常数 pi、指数函数等

from numpy.random import random
# 导入 numpy 库中的 random 函数，用于生成随机数

def direct_diff(x,k=1,period=None):
    # 对输入信号 x 进行傅里叶变换
    fx = fft(x)
    # 获取信号的长度
    n = len(fx)
    # 如果未指定周期，则默认为2*pi
    if period is None:
        period = 2*pi
    # 计算频率轴
    w = fftfreq(n)*2j*pi/period*n
    # 根据阶数 k 的正负，调整频率轴的幂次
    if k < 0:
        w = 1 / w**k
        w[0] = 0.0
    else:
        w = w**k
    # 如果信号长度大于2000，中心部分的频率成分置零
    if n > 2000:
        w[250:n-250] = 0.0
    # 对调整后的频率轴进行反傅里叶变换，并取实部得到差分结果
    return ifft(w*fx).real

def direct_tilbert(x,h=1,period=None):
    # 对输入信号 x 进行傅里叶变换
    fx = fft(x)
    # 获取信号的长度
    n = len(fx)
    # 如果未指定周期，则默认为2*pi
    if period is None:
        period = 2*pi
    # 计算频率轴
    w = fftfreq(n)*h*2*pi/period*n
    # 调整频率轴，使其符合 tilbert 变换的要求
    w[0] = 1
    w = 1j/tanh(w)
    w[0] = 0j
    # 对调整后的频率轴进行反傅里叶变换，得到 tilbert 变换的结果
    return ifft(w*fx)

def direct_itilbert(x,h=1,period=None):
    # 对输入信号 x 进行傅里叶变换
    fx = fft(x)
    # 获取信号的长度
    n = len(fx)
    # 如果未指定周期，则默认为2*pi
    if period is None:
        period = 2*pi
    # 计算频率轴
    w = fftfreq(n)*h*2*pi/period*n
    # 调整频率轴，使其符合 itilbert 变换的要求
    w = -1j*tanh(w)
    # 对调整后的频率轴进行反傅里叶变换，得到 itilbert 变换的结果
    return ifft(w*fx)

def direct_hilbert(x):
    # 对输入信号 x 进行傅里叶变换
    fx = fft(x)
    # 获取信号的长度
    n = len(fx)
    # 计算频率轴
    w = fftfreq(n)*n
    # 调整频率轴，使其符合 hilbert 变换的要求
    w = 1j*sign(w)
    # 对调整后的频率轴进行反傅里叶变换，得到 hilbert 变换的结果
    return ifft(w*fx)

def direct_ihilbert(x):
    # 对输入信号 x 进行 ihilbert 变换，即取负的 hilbert 变换结果
    return -direct_hilbert(x)

def direct_shift(x,a,period=None):
    # 获取信号的长度
    n = len(x)
    # 如果未指定周期，则默认使用单位周期
    if period is None:
        k = fftfreq(n)*1j*n
    else:
        k = fftfreq(n)*2j*pi/period*n
    # 对信号进行移频操作，然后进行反傅里叶变换得到结果
    return ifft(fft(x)*exp(k*a)).real

class TestDiff:
    # TestDiff 类，用于对差分函数的测试和验证
    # 定义一个测试方法 test_definition，用于测试 diff 和 direct_diff 函数
    def test_definition(self):
        # 遍历不同的 n 值进行测试
        for n in [16,17,64,127,32]:
            # 生成等差数组 x，长度为 n，用于计算 sin 和 cos 函数
            x = arange(n)*2*pi/n
            # 断言 diff(sin(x)) 的结果与 direct_diff(sin(x)) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x)), direct_diff(sin(x)))
            # 断言 diff(sin(x),2) 的结果与 direct_diff(sin(x),2) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),2), direct_diff(sin(x),2))
            # 断言 diff(sin(x),3) 的结果与 direct_diff(sin(x),3) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),3), direct_diff(sin(x),3))
            # 断言 diff(sin(x),4) 的结果与 direct_diff(sin(x),4) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),4), direct_diff(sin(x),4))
            # 断言 diff(sin(x),5) 的结果与 direct_diff(sin(x),5) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),5), direct_diff(sin(x),5))
            # 断言 diff(sin(2*x),3) 的结果与 direct_diff(sin(2*x),3) 的结果几乎相等
            assert_array_almost_equal(diff(sin(2*x),3), direct_diff(sin(2*x),3))
            # 断言 diff(sin(2*x),4) 的结果与 direct_diff(sin(2*x),4) 的结果几乎相等
            assert_array_almost_equal(diff(sin(2*x),4), direct_diff(sin(2*x),4))
            # 断言 diff(cos(x)) 的结果与 direct_diff(cos(x)) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x)), direct_diff(cos(x)))
            # 断言 diff(cos(x),2) 的结果与 direct_diff(cos(x),2) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x),2), direct_diff(cos(x),2))
            # 断言 diff(cos(x),3) 的结果与 direct_diff(cos(x),3) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x),3), direct_diff(cos(x),3))
            # 断言 diff(cos(x),4) 的结果与 direct_diff(cos(x),4) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x),4), direct_diff(cos(x),4))
            # 断言 diff(cos(2*x)) 的结果与 direct_diff(cos(2*x)) 的结果几乎相等
            assert_array_almost_equal(diff(cos(2*x)), direct_diff(cos(2*x)))
            # 断言 diff(sin(x*n/8)) 的结果与 direct_diff(sin(x*n/8)) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x*n/8)), direct_diff(sin(x*n/8)))
            # 断言 diff(cos(x*n/8)) 的结果与 direct_diff(cos(x*n/8)) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x*n/8)), direct_diff(cos(x*n/8)))
            # 遍历 k 的范围进行测试
            for k in range(5):
                # 断言 diff(sin(4*x),k) 的结果与 direct_diff(sin(4*x),k) 的结果几乎相等
                assert_array_almost_equal(diff(sin(4*x),k), direct_diff(sin(4*x),k))
                # 断言 diff(cos(4*x),k) 的结果与 direct_diff(cos(4*x),k) 的结果几乎相等
                assert_array_almost_equal(diff(cos(4*x),k), direct_diff(cos(4*x),k))

    # 定义一个测试方法 test_period，用于测试带周期参数的 diff 函数
    def test_period(self):
        # 遍历不同的 n 值进行测试
        for n in [17,64]:
            # 生成等差数组 x，长度为 n，用于计算 sin 函数
            x = arange(n)/float(n)
            # 断言 diff(sin(2*pi*x),period=1) 的结果与 2*pi*cos(2*pi*x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(2*pi*x),period=1), 2*pi*cos(2*pi*x))
            # 断言 diff(sin(2*pi*x),3,period=1) 的结果与 -(2*pi)**3*cos(2*pi*x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(2*pi*x),3,period=1), -(2*pi)**3*cos(2*pi*x))

    # 定义一个测试方法 test_sin，用于测试 sin 和 cos 函数的 diff 结果
    def test_sin(self):
        # 遍历不同的 n 值进行测试
        for n in [32,64,77]:
            # 生成等差数组 x，长度为 n，用于计算 sin 和 cos 函数
            x = arange(n)*2*pi/n
            # 断言 diff(sin(x)) 的结果与 cos(x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x)), cos(x))
            # 断言 diff(cos(x)) 的结果与 -sin(x) 的结果几乎相等
            assert_array_almost_equal(diff(cos(x)), -sin(x))
            # 断言 diff(sin(x),2) 的结果与 -sin(x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),2), -sin(x))
            # 断言 diff(sin(x),4) 的结果与 sin(x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(x),4), sin(x))
            # 断言 diff(sin(4*x)) 的结果与 4*cos(4*x) 的结果几乎相等
            assert_array_almost_equal(diff(sin(4*x)), 4*cos(4*x))
            # 断言 diff(sin(sin(x))) 的结果与 cos(x)*cos(sin(x)) 的结果几乎相等
            assert_array_almost_equal(diff(sin(sin(x))), cos(x)*cos(sin(x)))

    # 定义一个测试方法 test_expr，用于测试复杂表达式的 diff 结果
    def test_expr(self):
        # 遍历不同的 n 值进行测试（仅前五个值）
        for n in [64,77,100,128,256]:
            # 生成等差数组 x，长度为 n，用于计算复杂表达式 f 和其导数 df、ddf
            x = arange(n)*2*pi/n
            # 定义表达式 f
            f = sin(x)*cos(4*x)+exp(sin(3*x))
            # 定义表达式 f 的导数 df
            df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
            # 定义表达式 f 的二阶导数 ddf
            ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
                 - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
            # 计算 f 的一阶导数 d1，并断言其与 df 的结果几乎相等
            d1 = diff(f)
            assert_array_almost_equal(d1, df)
            # 断言 df 的一阶导数结果与 ddf 的结果几乎相等
            assert_array_almost_equal(diff(df), ddf)
            # 断言 f 的二阶导数结果与 ddf 的结果几乎相等
            assert_array_almost_equal(diff(f,2), ddf)
            # 断言 ddf 的逆向一阶导数结果与 df 的结果几乎相等
            assert_array_almost_equal(diff(ddf,-1), df)
    # 定义测试函数，用于测试大表达式的数值计算
    def test_expr_large(self):
        # 对于不同的 n 值循环测试
        for n in [2048,4096]:
            # 生成长度为 n 的等差数列乘以 2π/n
            x = arange(n)*2*pi/n
            # 计算复杂的函数 f
            f = sin(x)*cos(4*x)+exp(sin(3*x))
            # 计算 f 的一阶导数 df
            df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
            # 计算 f 的二阶导数 ddf
            ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
                 - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
            # 断言函数 diff(f) 计算结果与 df 数组几乎相等
            assert_array_almost_equal(diff(f),df)
            # 断言函数 diff(df) 计算结果与 ddf 数组几乎相等
            assert_array_almost_equal(diff(df),ddf)
            # 断言函数 diff(ddf,-1) 计算结果与 df 数组几乎相等
            assert_array_almost_equal(diff(ddf,-1),df)
            # 断言函数 diff(f,2) 计算结果与 ddf 数组几乎相等
            assert_array_almost_equal(diff(f,2),ddf)

    # 定义测试函数，用于测试整数 n 的情况
    def test_int(self):
        # 设置 n 的值为 64
        n = 64
        # 生成长度为 n 的等差数列乘以 2π/n
        x = arange(n)*2*pi/n
        # 断言函数 diff(sin(x),-1) 计算结果与 -cos(x) 数组几乎相等
        assert_array_almost_equal(diff(sin(x),-1),-cos(x))
        # 断言函数 diff(sin(x),-2) 计算结果与 -sin(x) 数组几乎相等
        assert_array_almost_equal(diff(sin(x),-2),-sin(x))
        # 断言函数 diff(sin(x),-4) 计算结果与 sin(x) 数组几乎相等
        assert_array_almost_equal(diff(sin(x),-4),sin(x))
        # 断言函数 diff(2*cos(2*x),-1) 计算结果与 sin(2*x) 数组几乎相等
        assert_array_almost_equal(diff(2*cos(2*x),-1),sin(2*x))

    # 定义测试函数，用于测试随机生成的偶数长度数组
    def test_random_even(self):
        # 对于不同的 k 和 n 值循环测试
        for k in [0,2,4,6]:
            for n in [60,32,64,56,55]:
                # 生成长度为 n 的随机数组 f
                f = random((n,))
                # 计算 f 的平均值 af
                af = sum(f,axis=0)/n
                # 将 f 减去其平均值 af
                f = f-af
                # 对 f 进行两次一阶差分操作
                f = diff(diff(f,1),-1)
                # 断言 f 的总和接近 0
                assert_almost_equal(sum(f,axis=0),0.0)
                # 断言函数 diff(diff(f,k),-k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                # 断言函数 diff(diff(f,-k),k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,-k),k),f)

    # 定义测试函数，用于测试随机生成的奇数长度数组
    def test_random_odd(self):
        # 对于不同的 k 和 n 值循环测试
        for k in [0,1,2,3,4,5,6]:
            for n in [33,65,55]:
                # 生成长度为 n 的随机数组 f
                f = random((n,))
                # 计算 f 的平均值 af
                af = sum(f,axis=0)/n
                # 将 f 减去其平均值 af
                f = f-af
                # 断言 f 的总和接近 0
                assert_almost_equal(sum(f,axis=0),0.0)
                # 断言函数 diff(diff(f,k),-k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                # 断言函数 diff(diff(f,-k),k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,-k),k),f)

    # 定义测试函数，用于测试随机生成数组后对 Nyquist 模式清零的情况
    def test_zero_nyquist(self):
        # 对于不同的 k 和 n 值循环测试
        for k in [0,1,2,3,4,5,6]:
            for n in [32,33,64,56,55]:
                # 生成长度为 n 的随机数组 f
                f = random((n,))
                # 计算 f 的平均值 af
                af = sum(f,axis=0)/n
                # 将 f 减去其平均值 af
                f = f-af
                # 对 f 进行两次一阶差分操作
                f = diff(diff(f,1),-1)
                # 断言 f 的总和接近 0
                assert_almost_equal(sum(f,axis=0),0.0)
                # 断言函数 diff(diff(f,k),-k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                # 断言函数 diff(diff(f,-k),k) 计算结果与 f 数组几乎相等
                assert_array_almost_equal(diff(diff(f,-k),k),f)
class TestTilbert:

    def test_definition(self):
        # 遍历不同的带宽和信号长度的组合
        for h in [0.1,0.5,1,5.5,10]:
            for n in [16,17,64,127]:
                # 生成等间距的角度数组
                x = arange(n)*2*pi/n
                # 对 sin(x) 应用 Tilbert 变换
                y = tilbert(sin(x),h)
                # 直接计算 Tilbert 变换
                y1 = direct_tilbert(sin(x),h)
                # 断言两种计算方式的结果近似相等
                assert_array_almost_equal(y,y1)
                # 断言 Tilbert 变换和其逆变换的结果近似相等
                assert_array_almost_equal(tilbert(sin(x),h),
                                          direct_tilbert(sin(x),h))
                # 断言对 sin(2*x) 应用 Tilbert 变换和其逆变换的结果近似相等
                assert_array_almost_equal(tilbert(sin(2*x),h),
                                          direct_tilbert(sin(2*x),h))

    def test_random_even(self):
        # 遍历不同的带宽和偶数长度的随机信号
        for h in [0.1,0.5,1,5.5,10]:
            for n in [32,64,56]:
                # 生成随机信号并去除其均值
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                # 断言去均值后的信号总和近似为零
                assert_almost_equal(sum(f,axis=0),0.0)
                # 断言直接计算逆 Tilbert 变换后再计算 Tilbert 变换得到原信号
                assert_array_almost_equal(direct_tilbert(direct_itilbert(f,h),h),f)

    def test_random_odd(self):
        # 遍历不同的带宽和奇数长度的随机信号
        for h in [0.1,0.5,1,5.5,10]:
            for n in [33,65,55]:
                # 生成随机信号并去除其均值
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                # 断言去均值后的信号总和近似为零
                assert_almost_equal(sum(f,axis=0),0.0)
                # 断言对随机信号应用 Tilbert 变换和逆 Tilbert 变换得到原信号
                assert_array_almost_equal(itilbert(tilbert(f,h),h),f)
                # 断言对随机信号应用 Tilbert 变换两次和逆 Tilbert 变换得到原信号
                assert_array_almost_equal(tilbert(itilbert(f,h),h),f)


class TestITilbert:

    def test_definition(self):
        # 遍历不同的带宽和信号长度的组合
        for h in [0.1,0.5,1,5.5,10]:
            for n in [16,17,64,127]:
                # 生成等间距的角度数组
                x = arange(n)*2*pi/n
                # 对 sin(x) 应用逆 Tilbert 变换
                y = itilbert(sin(x),h)
                # 直接计算逆 Tilbert 变换
                y1 = direct_itilbert(sin(x),h)
                # 断言两种计算方式的结果近似相等
                assert_array_almost_equal(y,y1)
                # 断言逆 Tilbert 变换和其逆变换的结果近似相等
                assert_array_almost_equal(itilbert(sin(x),h),
                                          direct_itilbert(sin(x),h))
                # 断言对 sin(2*x) 应用逆 Tilbert 变换和其逆变换的结果近似相等
                assert_array_almost_equal(itilbert(sin(2*x),h),
                                          direct_itilbert(sin(2*x),h))


class TestHilbert:

    def test_definition(self):
        # 遍历不同的信号长度
        for n in [16,17,64,127]:
            # 生成等间距的角度数组
            x = arange(n)*2*pi/n
            # 对 sin(x) 应用 Hilbert 变换
            y = hilbert(sin(x))
            # 直接计算 Hilbert 变换
            y1 = direct_hilbert(sin(x))
            # 断言两种计算方式的结果近似相等
            assert_array_almost_equal(y,y1)
            # 断言对 sin(2*x) 应用 Hilbert 变换和其逆变换的结果近似相等
            assert_array_almost_equal(hilbert(sin(2*x)),
                                      direct_hilbert(sin(2*x)))

    def test_tilbert_relation(self):
        # 遍历不同的信号长度
        for n in [16,17,64,127]:
            # 生成等间距的角度数组
            x = arange(n)*2*pi/n
            # 构造复合信号
            f = sin(x)+cos(2*x)*sin(x)
            # 对复合信号应用 Hilbert 变换
            y = hilbert(f)
            # 直接计算 Hilbert 变换
            y1 = direct_hilbert(f)
            # 断言两种计算方式的结果近似相等
            assert_array_almost_equal(y,y1)
            # 对复合信号应用 Tilbert 变换并断言与 Hilbert 变换结果近似相等
            y2 = tilbert(f,h=10)
            assert_array_almost_equal(y,y2)

    def test_random_odd(self):
        # 遍历不同的信号长度
        for n in [33,65,55]:
            # 生成随机信号并去除其均值
            f = random((n,))
            af = sum(f,axis=0)/n
            f = f-af
            # 断言去均值后的信号总和近似为零
            assert_almost_equal(sum(f,axis=0),0.0)
            # 断言对随机信号应用 Hilbert 变换和逆 Hilbert 变换得到原信号
            assert_array_almost_equal(ihilbert(hilbert(f)),f)
            # 断言对随机信号应用 Hilbert 变换两次和逆 Hilbert 变换得到原信号
            assert_array_almost_equal(hilbert(ihilbert(f)),f)
    # 定义测试函数 test_random_even，用于测试特定情况下的功能
    def test_random_even(self):
        # 遍历给定的数字列表 [32, 64, 56]
        for n in [32, 64, 56]:
            # 生成一个形状为 (n,) 的随机数组 f
            f = random((n,))
            # 计算 f 的平均值 af
            af = sum(f, axis=0) / n
            # 对 f 减去平均值 af，用于去除偏差
            f = f - af
            # 针对 Nyquist 模式进行归零处理:
            # 对 f 进行两次一阶差分，以消除 Nyquist 模式
            f = diff(diff(f, 1), -1)
            # 断言 f 的各列之和接近零
            assert_almost_equal(sum(f, axis=0), 0.0)
            # 断言直接希尔伯特变换的逆变换应返回原始数组 f
            assert_array_almost_equal(direct_hilbert(direct_ihilbert(f)), f)
            # 断言希尔伯特变换的逆变换应返回原始数组 f
            assert_array_almost_equal(hilbert(ihilbert(f)), f)
class TestIHilbert:

    def test_definition(self):
        # 针对不同的长度值进行测试
        for n in [16,17,64,127]:
            # 生成长度为 n 的等差数列，计算 x
            x = arange(n)*2*pi/n
            # 计算 sin(x) 的希尔伯特变换
            y = ihilbert(sin(x))
            # 使用直接方法计算 sin(x) 的希尔伯特变换
            y1 = direct_ihilbert(sin(x))
            # 断言两种计算结果近似相等
            assert_array_almost_equal(y,y1)
            # 断言 ihilbert(sin(2*x)) 与 direct_ihilbert(sin(2*x)) 的结果近似相等
            assert_array_almost_equal(ihilbert(sin(2*x)),
                                      direct_ihilbert(sin(2*x)))

    def test_itilbert_relation(self):
        # 针对不同的长度值进行测试
        for n in [16,17,64,127]:
            # 生成长度为 n 的等差数列，计算 x
            x = arange(n)*2*pi/n
            # 计算 sin(x) + cos(2*x)*sin(x) 的希尔伯特变换
            f = sin(x)+cos(2*x)*sin(x)
            # 计算 f 的希尔伯特变换
            y = ihilbert(f)
            # 使用直接方法计算 f 的希尔伯特变换
            y1 = direct_ihilbert(f)
            # 断言两种计算结果近似相等
            assert_array_almost_equal(y,y1)
            # 计算 f 的反希尔伯特变换，指定 h=10
            y2 = itilbert(f,h=10)
            # 断言希尔伯特变换与反希尔伯特变换结果近似相等
            assert_array_almost_equal(y,y2)


class TestShift:

    def test_definition(self):
        # 针对不同的长度值进行测试
        for n in [18,17,64,127,32,2048,256]:
            # 生成长度为 n 的等差数列，计算 x
            x = arange(n)*2*pi/n
            # 针对不同的平移值进行测试
            for a in [0.1,3]:
                # 断言 shift(sin(x),a) 与 direct_shift(sin(x),a) 的结果近似相等
                assert_array_almost_equal(shift(sin(x),a),direct_shift(sin(x),a))
                # 断言 shift(sin(x),a) 与 sin(x+a) 的结果近似相等
                assert_array_almost_equal(shift(sin(x),a),sin(x+a))
                # 断言 shift(cos(x),a) 与 cos(x+a) 的结果近似相等
                assert_array_almost_equal(shift(cos(x),a),cos(x+a))
                # 断言 shift(cos(2*x)+sin(x),a) 与 cos(2*(x+a))+sin(x+a) 的结果近似相等
                assert_array_almost_equal(shift(cos(2*x)+sin(x),a),
                                          cos(2*(x+a))+sin(x+a))
                # 断言 shift(exp(sin(x)),a) 与 exp(sin(x+a)) 的结果近似相等
                assert_array_almost_equal(shift(exp(sin(x)),a),exp(sin(x+a)))
            # 断言 shift(sin(x),2*pi) 与 sin(x) 的结果近似相等
            assert_array_almost_equal(shift(sin(x),2*pi),sin(x))
            # 断言 shift(sin(x),pi) 与 -sin(x) 的结果近似相等
            assert_array_almost_equal(shift(sin(x),pi),-sin(x))
            # 断言 shift(sin(x),pi/2) 与 cos(x) 的结果近似相等
            assert_array_almost_equal(shift(sin(x),pi/2),cos(x))


class TestOverwrite:
    """Check input overwrite behavior """

    real_dtypes = (np.float32, np.float64)
    dtypes = real_dtypes + (np.complex64, np.complex128)

    def _check(self, x, routine, *args, **kwargs):
        # 复制输入数组 x
        x2 = x.copy()
        # 调用指定的函数 routine，对 x2 进行操作
        routine(x2, *args, **kwargs)
        # 获取 routine 的名称作为签名
        sig = routine.__name__
        # 如果有传入参数，则将参数表示添加到签名中
        if args:
            sig += repr(args)
        # 如果有传入关键字参数，则将关键字参数表示添加到签名中
        if kwargs:
            sig += repr(kwargs)
        # 断言 x2 与原始输入 x 相等，检查是否存在不应该的覆盖
        assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

    def _check_1d(self, routine, dtype, shape, *args, **kwargs):
        np.random.seed(1234)
        # 如果是复数类型，则生成复数随机数数组；否则生成实数随机数数组
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        # 将数据类型转换为指定的 dtype
        data = data.astype(dtype)
        # 调用 _check 方法，检查对指定类型和形状的数据进行 routine 操作时的覆盖情况
        self._check(data, routine, *args, **kwargs)

    def test_diff(self):
        # 针对不同的数据类型进行测试
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，检查对长度为 16 的数据进行 diff 操作时的覆盖情况
            self._check_1d(diff, dtype, (16,))

    def test_tilbert(self):
        # 针对不同的数据类型进行测试
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，检查对长度为 16 的数据进行 tilbert 操作时的覆盖情况
            self._check_1d(tilbert, dtype, (16,), 1.6)

    def test_itilbert(self):
        # 针对不同的数据类型进行测试
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，检查对长度为 16 的数据进行 itilbert 操作时的覆盖情况
            self._check_1d(itilbert, dtype, (16,), 1.6)

    def test_hilbert(self):
        # 针对不同的数据类型进行测试
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，检查对长度为 16 的数据进行 hilbert 操作时的覆盖情况
            self._check_1d(hilbert, dtype, (16,))

    def test_cs_diff(self):
        # 针对不同的数据类型进行测试
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，检查对长度为 16 的数据进行 cs_diff 操作时的覆盖情况
            self._check_1d(cs_diff, dtype, (16,), 1.0, 4.0)
    # 定义测试方法，用于测试 sc_diff 函数
    def test_sc_diff(self):
        # 遍历 self.dtypes 中的每种数据类型
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，测试 sc_diff 函数在当前数据类型下的表现
            self._check_1d(sc_diff, dtype, (16,), 1.0, 4.0)

    # 定义测试方法，用于测试 ss_diff 函数
    def test_ss_diff(self):
        # 遍历 self.dtypes 中的每种数据类型
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，测试 ss_diff 函数在当前数据类型下的表现
            self._check_1d(ss_diff, dtype, (16,), 1.0, 4.0)

    # 定义测试方法，用于测试 cc_diff 函数
    def test_cc_diff(self):
        # 遍历 self.dtypes 中的每种数据类型
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，测试 cc_diff 函数在当前数据类型下的表现
            self._check_1d(cc_diff, dtype, (16,), 1.0, 4.0)

    # 定义测试方法，用于测试 shift 函数
    def test_shift(self):
        # 遍历 self.dtypes 中的每种数据类型
        for dtype in self.dtypes:
            # 调用 _check_1d 方法，测试 shift 函数在当前数据类型下的表现
            self._check_1d(shift, dtype, (16,), 1.0)
```